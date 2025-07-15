import copy
import warnings
import wandb

import torch
import transformers
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
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
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

from utils import (
    load_text_encoders,
    load_corgie_dataloader,
    load_removeObject_dataloader,
    remove_cache_and_checkpoints,
    collate_fn,
    compute_text_embeddings,
    log_validation,
    get_sigmas,
)
from huggingface_hub import snapshot_download

# Parameters
set_seed(42)
debug = False
pretrained_model_name_or_path = "black-forest-labs/FLUX.1-Fill-dev"
target_repo = "g-ronimo/flux-fill_ObjectRemoval-LoRA_6thTry"
device = "cuda"
weight_dtype = torch.bfloat16
learning_rate = 1e-4
num_epochs = 400
batch_size = 4
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
weighting_scheme = "none"
logit_mean = 0.0 
logit_std = 1.0
guidance_scale = 3.5
max_grad_norm = 1.0
instance_prompt = "Remove person or object."
resolution = 512

# Eval ..
validation_prompt = instance_prompt
num_validation_images = 1 
val_image = load_image("./validation_remove.jpg")
val_mask = load_image("./validation_remove_mask.png")

## FUNCTION defs

# save and upload LoRA adapter
def upload_adapter(model, target_repo, local_dir="./adapter"):
    # target_repo = "g-ronimo/flux-fill_LoRA_test"
    repo_id = create_repo(
        repo_id = target_repo,
        exist_ok = True,
    ).repo_id

    transformer_lora_layers_to_save = get_peft_model_state_dict(model)

    FluxFillPipeline.save_lora_weights(
        save_directory = local_dir,
        transformer_lora_layers = transformer_lora_layers_to_save,
    )

    upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
    )


# Load scheduler
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    pretrained_model_name_or_path, subfolder="scheduler"
)
noise_scheduler_copy = copy.deepcopy(noise_scheduler)

# Load text encoders
print("Loading tokenizers and text encoders")
tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two = load_text_encoders(pretrained_model_name_or_path,)

# Load VAE
print("Loading VAE")
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="vae",
)

# Move to GPU, otherwise we might run out of RAM
[model.to(device, dtype=weight_dtype) for model in [vae, text_encoder_one, text_encoder_two]]

# Load Transformer
print("Loading diffusion transformer")
transformer = FluxTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="transformer", 
    torch_dtype=weight_dtype
)
transformer.to(device, dtype=weight_dtype)

# We only train the additional adapter LoRA layers
[model.requires_grad_(False) for model in [transformer, vae, text_encoder_one, text_encoder_two]]

# Save VRAM
transformer.enable_gradient_checkpointing()

# Setup LoRA
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
optimizer = optimizer_class(
    [{"params": transformer_lora_parameters, "lr": learning_rate}],
    betas=(adam_beta1, adam_beta2),
    weight_decay=adam_weight_decay,
    eps=adam_epsilon,
)

# Setup LR Scheduler
lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
)

# Load Dataset
print("Loading dataset")

train_dataloader = load_removeObject_dataloader(batch_size, resolution)

# Encode prompts
# !! what prompts are encoded here?! 
instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
    instance_prompt, 
    [text_encoder_one, text_encoder_two], 
    [tokenizer_one, tokenizer_two],
    max_sequence_length = max_sequence_length
)

free_memory()

prompt_embeds = instance_prompt_hidden_states
pooled_prompt_embeds = instance_pooled_prompt_embeds
text_ids = instance_text_ids

# TODO: Cache latents
# latents_cache = []
# for batch in tqdm(train_dataloader, desc="Caching latents"):
#     with torch.no_grad():
#         batch["pixel_values"] = batch["pixel_values"].to(
#             device, non_blocking=True, dtype=transformer.dtype
#         )
#         latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

# Setup Logging 
wandb.init(
    project="FLUX-fill LoRA", 
).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

# Prepare for validation
pipeline_args = {"prompt": validation_prompt, "image": val_image, "mask_image": val_mask}
pipeline = FluxFillPipeline(
    scheduler = noise_scheduler,
    vae = vae,
    text_encoder = text_encoder_one,
    text_encoder_2 = text_encoder_two,
    tokenizer = tokenizer_one,
    tokenizer_2 = tokenizer_two,
    transformer = transformer
)

# TRAIN!
global_step = 0

for epoch in range(num_epochs):
    transformer.train()
        
    for step, batch in enumerate(train_dataloader):
        prompts = batch["prompts"]

        # model_input = latents_cache[step].sample()
        # model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        # model_input = model_input.to(transformer.dtype)
        model_input = vae.encode(
            batch["images"].to(device).reshape(batch["images"].shape).to(transformer.dtype)
        ).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_input = model_input.to(transformer.dtype)
    
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        
        masked_image_latents = vae.encode(
            batch["images_masked"].to(device).reshape(batch["images"].shape).to(transformer.dtype)
        ).latent_dist.sample()
        
        masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
        
        mask = batch["masks"]    
        mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(
            model_input.shape[0], model_input.shape[2], vae_scale_factor, model_input.shape[3], vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            model_input.shape[0], vae_scale_factor * vae_scale_factor, model_input.shape[2], model_input.shape[3]
        )  # ba
        
        latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            device,
            transformer.dtype,
        )
        
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        
        # TODO: Simplify this, we don't the scheduler to sample a timestep!
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=logit_mean,
            logit_std=logit_std,
        )
        
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        
        sigmas = get_sigmas(timesteps, noise_scheduler_copy, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        
        packed_noisy_model_input = FluxFillPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
        if transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None
        
        masked_image_latents = FluxFillPipeline._pack_latents(
            masked_image_latents,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
        mask = FluxFillPipeline._pack_latents(
            mask,
            batch_size=model_input.shape[0],
            num_channels_latents=vae_scale_factor*vae_scale_factor,
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        masked_image_latents = torch.cat((masked_image_latents.to(device), mask.to(device)), dim=-1)
        
        transformer_input = torch.cat((packed_noisy_model_input, masked_image_latents), dim=2)    
        
        if debug:
            print("timesteps")
            print(timesteps)
            print("sigmas")
            print(sigmas)

        model_pred = transformer(
            hidden_states=transformer_input.to(transformer.dtype),
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        
        model_pred = FluxFillPipeline._unpack_latents(
            model_pred,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        target = noise - model_input
        
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        
        loss = loss.mean()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
            
        print(f"step {global_step}, epoch {epoch}, loss {loss.detach().item()}, grad_norm: {grad_norm}")
        
        logs = dict(
            step = step,
            epoch = epoch,
            loss = loss.detach().item(), 
            lr = lr_scheduler.get_last_lr()[0], 
            grad_norm = grad_norm,
        )
        wandb.log(logs)

        if global_step % 100 == 0:
            images = log_validation(
                pipeline=pipeline,
                pipeline_args=pipeline_args,
                epoch=epoch,
                validation_prompt=validation_prompt,
                num_validation_images=num_validation_images,
             )
            wandb.log(dict(validation = [ wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images) ]))
    
        global_step += 1

print("Uploading adapter")
upload_adapter(transformer, target_repo)

wandb.finish()

import os
os.system('runpodctl remove pod $RUNPOD_POD_ID')


