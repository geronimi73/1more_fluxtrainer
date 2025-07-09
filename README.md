minimal FLUX1-dev fill LoRA training script

# Motivation
the latest "fill" variant of the FLUX1 family of models is powerful. The few training scripts I found are too complicated for my simple brain. Here I am trying to understand what's happening and distilling a minimal script

# Usage
not yet 

# Todo
- [x] make it work with Corgies
- [ ] save VRAM, make it work on 24GB GPU: 4bit transformer? offload VAE? encode prompts and offload text encoders?
- [ ] add custom dataset
- [x] refactor dataset class
- [ ] figure out what all this code actually does
- [ ] save checkpoints
- [ ] refactor refactor refactor
- [ ] train a proper LoRA that actually does something useful
