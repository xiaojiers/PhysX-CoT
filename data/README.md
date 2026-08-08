# Data layout

The public repository does not redistribute PhysX-CoT training or evaluation
data. The launch scripts use the following conventional layout when no
environment-variable override is supplied:

```text
data/
  train.jsonl
  renders/
  sam_features/
  cache/
```

GRPO additionally requires per-part voxel targets referenced by the generated
training JSONL. Dataset URLs, checksums, and license terms must be added before
the reproducibility release.
