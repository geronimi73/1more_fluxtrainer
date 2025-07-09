import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
# from PIL import Image, ImageDraw
# from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import set_seed

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxFillPipeline,
    FluxTransformer2DModel,
)
from diffusers.utils import load_image
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from utils import (
    load_text_encoders,
    DreamBoothDataset,
    remove_cache_and_checkpoints,
    collate_fn,
    compute_text_embeddings,
    get_mask,
    prepare_mask_and_masked_image,
)
from huggingface_hub import snapshot_download

# Parameters
set_seed(42)
pretrained_model_name_or_path = "black-forest-labs/FLUX.1-Fill-dev"
device = "cuda"
weight_dtype = torch.bfloat16
learning_rate = 1e-4
lr_warmup_steps = 100
max_train_steps = 1_000
lr_num_cycles = 1
lr_power = 1.0
gradient_accumulation_steps = 1 
train_batch_size = 1 
rank = 4
alpha = 4
target_modules = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]
optimizer_class = torch.optim.AdamW
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-04
adam_epsilon = 1e-08
max_sequence_length = 512
validation_prompt="A TOK dog"
weighting_scheme = "none"
logit_mean = 0.0 
logit_std = 1.0
guidance_scale = 3.5
max_grad_norm = 1.0
# mode_scale = None

# Load scheduler
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    pretrained_model_name_or_path, subfolder="scheduler"
)
noise_scheduler_copy = copy.deepcopy(noise_scheduler)

# Load text encoders
print("Loading tokenizers and text encoders")
tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two = load_text_encoders(pretrained_model_name_or_path,)
tokenizers = [tokenizer_one, tokenizer_two]
text_encoders = [text_encoder_one, text_encoder_two]

# Load VAE
print("Loading VAE")
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="vae",
)
vae_config_shift_factor = vae.config.shift_factor
vae_config_scaling_factor = vae.config.scaling_factor
vae_config_block_out_channels = vae.config.block_out_channels

# Move to GPU, otherwise we might run out of RAM
vae.to(device, dtype=weight_dtype)
text_encoder_one.to(device, dtype=weight_dtype)
text_encoder_two.to(device, dtype=weight_dtype)

# Load Transformer
print("Loading diffusion transformer")
transformer = FluxTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="transformer", 
    torch_dtype=weight_dtype
)
transformer.to(device, dtype=weight_dtype)

# We only train the additional adapter LoRA layers
transformer.requires_grad_(False)
vae.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

transformer.enable_gradient_checkpointing()
# text_encoder_one.gradient_checkpointing_enable()

# # Setup LoRA
print("Adding adapters")
transformer_lora_config = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    init_lora_weights="gaussian",
    target_modules=target_modules,
)
transformer.add_adapter(transformer_lora_config)
transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

# Setup Optimizer
transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": learning_rate}
params_to_optimize = [transformer_parameters_with_lr]

optimizer = optimizer_class(
    params_to_optimize,
    betas=(adam_beta1, adam_beta2),
    weight_decay=adam_weight_decay,
    eps=adam_epsilon,
)

# Setup LR Scheduler
lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps,
    num_training_steps=max_train_steps,
    num_cycles=lr_num_cycles,
    power=lr_power,
)

# Load Dataset
print("Loading dataset")
snapshot_download(
    "diffusers/dog-example",
    local_dir= "./dog", repo_type="dataset",
    ignore_patterns=".gitattributes",
)

snapshot_download(
    "sebastianzok/dog-example-masks",
    local_dir= "./dog_masks", repo_type="dataset",
    ignore_patterns=".gitattributes",
)

[remove_cache_and_checkpoints(d) for d in ["./dog", "./dog_masks"]]


instance_data_dir = "dog"
mask_data_dir = "dog_masks"
instance_prompt = "A TOK dog"
# class_prompt = None
# num_class_images = 100
resolution = 512
repeats = 1 
# center_crop = False

train_dataset = DreamBoothDataset(
    instance_data_root=instance_data_dir,
    mask_data_root=mask_data_dir,
    instance_prompt=instance_prompt,
    class_prompt=None,
    class_data_root=None,
    # class_num=num_class_images,
    size=resolution,
    repeats=repeats,
    center_crop=False,
    resolution = resolution,
)

# Load DataLoader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, with_prior_preservation=False),
    # num_workers=args.dataloader_num_workers,
)

# Encode prompts
instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
    instance_prompt, text_encoders, tokenizers,
    max_sequence_length = max_sequence_length
)

del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
free_memory()

prompt_embeds = instance_prompt_hidden_states
pooled_prompt_embeds = instance_pooled_prompt_embeds
text_ids = instance_text_ids

# Cache latents
latents_cache = []
for batch in tqdm(train_dataloader, desc="Caching latents"):
    with torch.no_grad():
        batch["pixel_values"] = batch["pixel_values"].to(
            device, non_blocking=True, dtype=weight_dtype
        )
        latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

if validation_prompt is None:
    del vae
    free_memory()

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
num_update_steps_per_epoch

num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
num_train_epochs
