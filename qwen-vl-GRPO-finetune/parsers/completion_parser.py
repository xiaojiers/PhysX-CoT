"""
parsers/completion_parser.py

V3          SFT    5generate_cot_finetune2.py       

   completion       

    <think>
    Step 1: ...      part_count: <N>
    Step 2: ...      bbox_2d=[xmin,xmax,ymin,ymax],
                     bbox_3d=[xmin,xmax,ymin,ymax,zmin,zmax],
                     sam_feat=<sam_feat_l_k>
    Step 3: ...      Part `l_i`: `l_j` is at ['top'], `l_k` is at ['bottom'].
    Step 4: ...      shape_label=<...>, major_axis=<...>, aspect_ratio=<...>
    Step 5: ...      hardness=<...>, roughness=<...>, reflectivity=<...>, transparency=<...>
    </think>

    <overall>
    Name: ...
    Category: ...
    Dimension: a*b*c
    Parts:
    l_<id>: <part_name>, <aff_rank>, <material>, <density> g/cm^3, <young>, <poisson>, <desc>
    ...
    Group_info:
    group_<id>: ['l_x', ...]; Type: <E|R|P|S|H|F>; Param: ...
    </overall>

    <geometry_l_k>
    0 1-5 36-41 ...
    </geometry_l_k>

    
    CompletionParser.parse(text)   dict think_steps / overall_dict / geometries / raw 
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from configs import TaskConfig

LOGGER = logging.getLogger(__name__)


#                                                                
#                            
#                                                                

# Step 1: part_count
_PAT_PART_COUNT = re.compile(r"part_count[^:]*:\s*(\d+)", re.IGNORECASE)
MAX_PARTS = 24

# Step 2:    Part      bbox_2d / bbox_3d / sam_feat      
_PAT_PART_BLOCK = re.compile(
    r"Part\s*`l_(\d+)`(.*?)(?=Part\s*`l_\d+`|Step\s*\d|</think>|<overall>|$)",
    re.DOTALL | re.IGNORECASE,
)
_PAT_BBOX2D = re.compile(
    r"`?bbox_2d`?\s*=\s*\[\s*([\d.\-eE+,\s]+?)\s*\]"
)
_PAT_BBOX3D = re.compile(
    r"`?bbox_3d`?\s*=\s*\[\s*([\d\-,\s]+?)\s*\]"
)

# Step 3: "Part `l_i`: `l_j` is at ['top', ...]., `l_k` is at ['bottom']..."
_PAT_INTER_BLOCK = re.compile(
    r"Part\s*`l_(\d+)`\s*:\s*(.*?)(?=Part\s*`l_\d+`\s*:|Step\s*\d|</think>|<overall>|$)",
    re.DOTALL | re.IGNORECASE,
)
_PAT_NEIGHBOR = re.compile(
    r"`?l_(\d+)`?\s*is\s*at\s*\[\s*([^\]]*?)\s*\]",
    re.IGNORECASE,
)

# Step 4: primitive        Part    
_PAT_PRIMITIVE_BLOCK = re.compile(
    r"Part\s*`l_(\d+)`(.*?)(?=Part\s*`l_\d+`|Step\s*\d|</think>|<overall>|$)",
    re.DOTALL | re.IGNORECASE,
)
_PAT_SHAPE_LABEL = re.compile(r"`?shape_label`?\s*=\s*([A-Za-z_]+)")
_PAT_MAJOR_AXIS  = re.compile(r"`?major_axis`?\s*=\s*([A-Za-z_]+)")
_PAT_ASPECT      = re.compile(r"`?aspect_ratio`?\s*=\s*([A-Za-z_]+)")

# Step 5: surface    
_PAT_SURFACE = re.compile(
    r"Part\s*`l_(\d+)`[^`]*"
    r"`?hardness`?\s*=\s*([A-Za-z_]+)\s*,\s*"
    r"`?roughness`?\s*=\s*([A-Za-z_]+)\s*,\s*"
    r"`?reflectivity`?\s*=\s*([A-Za-z_]+)\s*,\s*"
    r"`?transparency`?\s*=\s*([A-Za-z_]+)",
    re.IGNORECASE,
)

# Geometry: <geometry_l_k> ... </geometry_l_k>
_PAT_GEOMETRY = re.compile(
    r"<geometry_l_(\d+)>\s*(.*?)\s*</geometry_l_(\d+)>",
    re.DOTALL,
)

# Overall      
_PAT_OVERALL_NAME      = re.compile(r"^\s*Name\s*:\s*(.+)$",      re.MULTILINE | re.IGNORECASE)
_PAT_OVERALL_CATEGORY  = re.compile(r"^\s*Category\s*:\s*(.+)$",  re.MULTILINE | re.IGNORECASE)
_PAT_OVERALL_DIMENSION = re.compile(
    r"^\s*Dimension\s*:\s*([\d.]+)\s*\*\s*([\d.]+)\s*\*\s*([\d.]+)",
    re.MULTILINE | re.IGNORECASE,
)
_PAT_OVERALL_PART = re.compile(
    r"^\s*l_(\d+)\s*:\s*"
    r"([^,]+),\s*"
    r"(\d+),\s*"
    r"([^,]+),\s*"
    r"([\d.]+)\s*g/cm\^3,\s*"
    r"([\d.]+),\s*"
    r"([\d.]+),\s*"
    r"(.+)$",
    re.MULTILINE,
)
_PAT_OVERALL_GROUP = re.compile(
    r"^\s*group_(\d+)\s*:\s*\[([^\]]+)\]\s*;\s*"
    r"Type\s*:\s*([^;]+);\s*"
    r"Params?\s*:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


#                                                                
#    
#                                                                

class CompletionParser:

    def __init__(self, task_cfg: TaskConfig):
        self.task_cfg = task_cfg

    #                                                                       

    def extract_sections(self, text: str) -> Dict[str, Optional[str]]:
        """
          completion          
          think    : <think>...</think>
          overall  : <overall>...</overall>      V3     
                       </think>               SFT    
          final    : <final>...</final>              GRPO     
        Geometry      parse_geometries          
        """
        out: Dict[str, Optional[str]] = {"think": None, "overall": None, "final": None}

        t_open  = self.task_cfg.think_open_tag
        t_close = self.task_cfg.think_close_tag
        if t_open in text and t_close in text:
            seg = text.split(t_open, 1)[1].split(t_close, 1)[0]
            out["think"] = seg.strip()
            after = text.split(t_close, 1)[1]
        else:
            after = text

        o_open  = self.task_cfg.overall_open_tag
        o_close = self.task_cfg.overall_close_tag
        if o_open in after and o_close in after:
            out["overall"] = after.split(o_open, 1)[1].split(o_close, 1)[0].strip()
        else:
            f_open  = self.task_cfg.final_open_tag
            f_close = self.task_cfg.final_close_tag
            if f_open in after and f_close in after:
                out["final"] = after.split(f_open, 1)[1].split(f_close, 1)[0].strip()
            else:
                #     SFT    </think>        
                stripped = after.strip()
                if stripped and ("Name:" in stripped or "Parts:" in stripped):
                    out["overall"] = stripped

        return out

    #    think 5-step                                                        

    def parse_think_steps(self, think_text: Optional[str]) -> Dict[str, Any]:
        """
          <think>            

        Returns
        -------
        dict        None / {} / set()  
            steps_found      : Set[int]                      Step      {1..5}
            part_count       : Optional[int]
            bbox_2d          : Dict[str, [xmin,xmax,ymin,ymax]]
            bbox_3d          : Dict[str, [xmin,xmax,ymin,ymax,zmin,zmax]]
            inter_part_positions : Dict[str, Dict[str, List[str]]]
                                  {l_i: {l_j: [directions...], ...}, ...}
            primitive_shape  : Dict[str, {shape_label, major_axis, aspect_ratio}]
            surface_features : Dict[str, {hardness, roughness, reflectivity, transparency}]
        """
        result: Dict[str, Any] = {
            "steps_found":          set(),
            "part_count":           None,
            "bbox_2d":              {},
            "bbox_3d":              {},
            "inter_part_positions": {},
            "primitive_shape":      {},
            "surface_features":     {},
        }
        if not think_text:
            return result

        #    Step 1: part_count                                  
        if "Step 1" in think_text or "step 1" in think_text:
            result["steps_found"].add(1)
        m = _PAT_PART_COUNT.search(think_text)
        if m:
            part_count = int(m.group(1))
            if 1 <= part_count <= MAX_PARTS:
                result["part_count"] = part_count

        #    Step 2   bbox_2d / bbox_3d                       
        step2_seg = self._slice_step(think_text, 2)
        if step2_seg is not None:
            result["steps_found"].add(2)
            for part_m in _PAT_PART_BLOCK.finditer(step2_seg):
                pid = f"l_{part_m.group(1)}"
                blob = part_m.group(2)
                b2 = self._parse_float_list(_PAT_BBOX2D.search(blob), expect=4)
                if b2 is not None:
                    result["bbox_2d"][pid] = b2
                b3 = self._parse_int_list(_PAT_BBOX3D.search(blob), expect=6)
                if b3 is not None:
                    result["bbox_3d"][pid] = b3

        #    Step 3   inter_part_positions                     
        step3_seg = self._slice_step(think_text, 3)
        if step3_seg is not None:
            result["steps_found"].add(3)
            for part_m in _PAT_INTER_BLOCK.finditer(step3_seg):
                pid = f"l_{part_m.group(1)}"
                blob = part_m.group(2)
                neighbors: Dict[str, List[str]] = {}
                for nb_m in _PAT_NEIGHBOR.finditer(blob):
                    nb_id = f"l_{nb_m.group(1)}"
                    raw_dirs = nb_m.group(2)
                    dirs = self._parse_direction_list(raw_dirs)
                    if dirs:
                        neighbors[nb_id] = dirs
                #             part       no adjacent parts 
                result["inter_part_positions"][pid] = neighbors

        #    Step 4   primitive                             
        step4_seg = self._slice_step(think_text, 4)
        if step4_seg is not None:
            result["steps_found"].add(4)
            for part_m in _PAT_PRIMITIVE_BLOCK.finditer(step4_seg):
                pid = f"l_{part_m.group(1)}"
                blob = part_m.group(2)
                shape_m = _PAT_SHAPE_LABEL.search(blob)
                axis_m  = _PAT_MAJOR_AXIS.search(blob)
                ratio_m = _PAT_ASPECT.search(blob)
                if shape_m or axis_m or ratio_m:
                    result["primitive_shape"][pid] = {
                        "shape_label":  shape_m.group(1).lower() if shape_m else None,
                        "major_axis":   axis_m.group(1).lower()  if axis_m  else None,
                        "aspect_ratio": ratio_m.group(1).lower() if ratio_m else None,
                    }

        #    Step 5   surface_features                         
        step5_seg = self._slice_step(think_text, 5)
        if step5_seg is not None:
            result["steps_found"].add(5)
            for sm in _PAT_SURFACE.finditer(step5_seg):
                pid = f"l_{sm.group(1)}"
                result["surface_features"][pid] = {
                    "hardness":     sm.group(2).lower(),
                    "roughness":    sm.group(3).lower(),
                    "reflectivity": sm.group(4).lower(),
                    "transparency": sm.group(5).lower(),
                }

        return result

    #    overall                                                            

    def parse_overall(self, overall_text: Optional[str]) -> Optional[Dict[str, Any]]:
        """
           <overall>         
              Name    parts    None 
        """
        if not overall_text or not overall_text.strip():
            return None

        result: Dict[str, Any] = {
            "name":       None,
            "category":   None,
            "dimension":  None,
            "parts":      {},
            "part_ids":   [],
            "group_info": [],
        }

        m = _PAT_OVERALL_NAME.search(overall_text)
        if m:
            result["name"] = m.group(1).strip()
        m = _PAT_OVERALL_CATEGORY.search(overall_text)
        if m:
            result["category"] = m.group(1).strip()
        m = _PAT_OVERALL_DIMENSION.search(overall_text)
        if m:
            result["dimension"] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]

        for pm in _PAT_OVERALL_PART.finditer(overall_text):
            pid = f"l_{pm.group(1)}"
            result["parts"][pid] = {
                "part_name":      pm.group(2).strip(),
                "material_index": int(pm.group(3)),
                "material":       pm.group(4).strip(),
                "density":        float(pm.group(5)),
                "young_modulus":  float(pm.group(6)),
                "poisson_ratio":  float(pm.group(7)),
                "description":    pm.group(8).strip(),
            }
            result["part_ids"].append(pid)
        result["part_ids"].sort(key=lambda x: int(x[2:]))

        for gm in _PAT_OVERALL_GROUP.finditer(overall_text):
            members = [
                p.strip().strip("'\"")
                for p in gm.group(2).split(",")
                if p.strip()
            ]
            result["group_info"].append({
                "id":     f"group_{gm.group(1)}",
                "parts":  members,
                "type":   gm.group(3).strip(),
                "param":  gm.group(4).strip(),
            })

        if not result["name"] and not result["parts"]:
            return None
        return result

    #    geometry_l_k                                                        

    def parse_geometries(self, text: str) -> Dict[int, str]:
        """     <geometry_l_k>...</geometry_l_k>      {k:    RLE    } """
        out: Dict[int, str] = {}
        for gm in _PAT_GEOMETRY.finditer(text):
            try:
                k_open  = int(gm.group(1))
                k_close = int(gm.group(3))
                if k_open != k_close:
                    continue
                if 0 <= k_open < MAX_PARTS:
                    out[k_open] = gm.group(2).strip()
            except ValueError:
                continue
        return out

    #                                                                       

    def parse(self, completion_text: str) -> Dict[str, Any]:
        sections = self.extract_sections(completion_text)
        think_steps  = self.parse_think_steps(sections["think"])
        overall_dict = self.parse_overall(sections["overall"])
        geometries   = self.parse_geometries(completion_text)

        return {
            "raw_text":      completion_text,
            "think_text":    sections["think"],
            "overall_text":  sections["overall"],
            "final_text":    sections["final"],
            "think_steps":   think_steps,
            "overall_dict":  overall_dict,
            "geometries":    geometries,        # Dict[int, str]
        }

    #                                                                
    #     
    #                                                                

    @staticmethod
    def _slice_step(think_text: str, step_no: int) -> Optional[str]:
        """  <think>       'Step N'      'Step N+1' /         """
        opens  = [m.start() for m in re.finditer(rf"Step\s*{step_no}\b", think_text, re.IGNORECASE)]
        if not opens:
            return None
        start = opens[0]
        nxt   = re.search(rf"Step\s*{step_no + 1}\b", think_text[start:], re.IGNORECASE)
        end   = start + nxt.start() if nxt else len(think_text)
        return think_text[start:end]

    @staticmethod
    def _parse_float_list(match, expect: int) -> Optional[List[float]]:
        if match is None:
            return None
        try:
            vals = [float(v.strip()) for v in match.group(1).split(",") if v.strip()]
        except ValueError:
            return None
        return vals if len(vals) == expect else None

    @staticmethod
    def _parse_int_list(match, expect: int) -> Optional[List[int]]:
        if match is None:
            return None
        try:
            vals = [int(round(float(v.strip()))) for v in match.group(1).split(",") if v.strip()]
        except ValueError:
            return None
        return vals if len(vals) == expect else None

    @staticmethod
    def _parse_direction_list(raw: str) -> List[str]:
        """  "'top', 'front', 'bottom'" / 'top, front, bottom'      ['top','front','bottom'] """
        if not raw:
            return []
        cleaned = raw.replace("'", " ").replace('"', " ").replace("`", " ")
        parts = [p.strip().lower() for p in cleaned.split(",") if p.strip()]
        return parts
