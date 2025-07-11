minimal FLUX1-dev fill LoRA training script

# Motivation
The few FLUX fill training scripts I found are too complicated for my simple brain. Here I am trying to understand what's happening and distilling a minimal training script

# Usage
not yet 

# Todo
- [x] make it work with Corgies
- [ ] save VRAM, make it work on 24GB GPU: 4bit transformer? offload VAE? encode prompts and offload text encoders?
- [x] add custom dataset
- [x] refactor dataset class
- [ ] figure out what all this code actually does
- [x] save checkpoints
- [ ] refactor refactor refactor
- [ ] train a proper LoRA that actually does something useful
