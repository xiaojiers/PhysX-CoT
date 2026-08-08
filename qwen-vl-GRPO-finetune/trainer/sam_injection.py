"""
trainer/sam_injection.py

SAM            + embed_tokens forward hook 

  SFT (qwen-vl-SFT-finetune/train_lora.py)
   GRPO      sam_projector.pt        /    

       SFT     
  1. SAMProjector   SAM3 RoI      [256]     LLM      [3584] 
  2. SAMEmbeddingInjector     model.get_input_embeddings()   forward hook 
         embed_tokens(input_ids)           input_ids  
     <sam_feat>     embedding         SAM    
  3.      <sam_feat>       l_0, l_1, ...      
       npz      part_id             

     GRPO         
    forward      injector.set_batch(sam_feats: Tensor[B, max_parts, D_sam])
    generate     injector.set_batch(sam_feats[:, :, :]    num_generations  )
    sam_feats=None   hook           
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


#                                                                            


class SAMProjector(nn.Module):
    """SAM RoI      [256]   [hidden_dim]   [llm_dim] 

        SFT      Linear   GELU   Linear         sam_projector.pt 
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 3584):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#    Forward Hook embed_tokens       <sam_feat>                         


class SAMEmbeddingInjector:
    """embed_tokens   forward hook   <sam_feat>       SAM      

      SFT     
        GRPO   rollout generate   compute_loss         forward 
           set_batch              
        num_generations > 1      prompt      completion  
        SAM     per-prompt       sam_feats   batch     
        num_generations     [B*G, max_parts, D_sam]   input_ids    
    """

    def __init__(self, projector: SAMProjector, sam_token_id: int):
        self.projector = projector
        self.sam_token_id = int(sam_token_id)
        self._pending_feats: Optional[torch.Tensor] = None
        #          pending True=      False=     forward    
        #    generate      forward 
        self._auto_clear = True

    #                                                                         

    def set_batch(self, sam_feats: Optional[torch.Tensor], auto_clear: bool = True) -> None:
        """        batch   SAM    

        Args:
          sam_feats : [B, max_parts, D_sam] float32 B        
                      embed_tokens   input_ids batch        num_generations     
                      None      batch   SAM    
          auto_clear: True       forward       pending 
                      False      pending    generate     forward         
        """
        self._pending_feats = sam_feats
        self._auto_clear = bool(auto_clear)

    def clear(self) -> None:
        self._pending_feats = None

    #    Hook                                                                 

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,
    ) -> torch.Tensor:
        sam_feats = self._pending_feats
        if sam_feats is None:
            return output

        input_ids = inputs[0]                               # [B, L]
        if input_ids is None or input_ids.dim() != 2:
            # generate       prompt-only / token-by-token    shape 
            # token-by-token (decode   )     <sam_feat>      
            return output

        if self._auto_clear:
            self._pending_feats = None

        #    batch        
        # trl GRPOTrainer   rollout       prompt    num_generations  
        #  repeat_interleave prompt0,prompt0,prompt1,prompt1,...  
        #      sam_feats   batch    < input_ids   batch    
        #               repeat_interleave    
        B_in   = input_ids.size(0)
        B_feat = sam_feats.size(0)
        if B_feat != B_in:
            if B_in % B_feat == 0:
                sam_feats = sam_feats.repeat_interleave(B_in // B_feat, dim=0)
            else:
                # batch                        
                return output

        projected = self.projector(
            sam_feats.to(device=output.device, dtype=output.dtype)
        )                                                   # [B, max_parts, D_llm]

        output = output.clone()
        for b in range(input_ids.size(0)):
            positions = (input_ids[b] == self.sam_token_id).nonzero(as_tuple=True)[0]
            n = min(positions.numel(), projected.size(1))
            if n > 0:
                output[b] = output[b].index_copy(0, positions[:n], projected[b, :n])
        return output


#         [B, P, D] sam_feats   batch      G   rollout             


def repeat_for_generations(
    sam_feats: Optional[torch.Tensor], num_generations: int
) -> Optional[torch.Tensor]:
    """  [B, P, D]   sam_feats   batch       num_generations   

    GRPO   generate      prompt    G   completion    model.generate
      input_ids   [B*G, L]       sam_feats     [B*G, P, D] 

    Args:
      sam_feats        : [B, P, D]   None
      num_generations  : G

    Returns:
      [B*G, P, D]   None
    """
    if sam_feats is None or num_generations <= 1:
        return sam_feats
    return sam_feats.repeat_interleave(num_generations, dim=0)


#         SFT checkpoint    sam_projector.pt                              


def load_sam_projector_state(projector: SAMProjector, ckpt_path: str) -> None:
    """  SFT     sam_projector.pt      

    SFT      torch.save(sam_projector.state_dict(), 'sam_projector.pt')
            load_state_dict strict=True    
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    missing, unexpected = projector.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SAMProjector load mismatch from {ckpt_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
