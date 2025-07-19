import copy, wandb, torch, os
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from fire import Fire

from huggingface_hub import create_repo, HfApi
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import set_seed
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
    get_maskeddataset_dataloader,
    remove_cache_and_checkpoints,
    collate_fn,
    compute_text_embeddings,
    get_sigmas,
    load_images_and_masks_from_folder,
    create_image_gallery,
)
from huggingface_hub import snapshot_download

# TODOs:
# * cache latents
# * load instance_prompt from dataset!

# try and create target repo
def create_target_repo(target_repo):
    repo_id = create_repo(
        repo_id = target_repo,
        exist_ok = True,
    ).repo_id

# save and upload LoRA adapter
def upload_adapter(model, target_repo, target_fn, local_dir="adapter"):
    print(f"Uploading adapter to {target_repo}")

    # Save adapter locally
    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
    FluxFillPipeline.save_lora_weights(
        save_directory = f"{local_dir}/{target_fn}",
        transformer_lora_layers = transformer_lora_layers_to_save,
    )

    # Upload to HF
    api = HfApi()
    api.upload_file(
        path_or_fileobj = f"{local_dir}/{target_fn}/pytorch_lora_weights.safetensors",
        path_in_repo=f"{target_fn}.safetensors",
        repo_id=target_repo,
    )

def run_validation(
    pipeline,
    validation_prompt,
    seed = 42,
    num_steps = 50,
):
    print(
        f"Running validation with prompt:"
        f" {validation_prompt}."
    )

    # run inference
    generator = torch.Generator(device=pipeline.device).manual_seed(seed) if seed else None

    images = []
    for val_image, val_mask in load_images_and_masks_from_folder("removal_validations"):
        pipeline_args = dict(
            prompt = validation_prompt,
            image = val_image,
            mask_image = val_mask,
            num_inference_steps = num_steps,
        )
        image = pipeline(**pipeline_args, generator=generator).images[0]
        images.append(image)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images

def main(
    target_repo: str = "g-ronimo/flux-fill_ObjectRemoval-LoRA_12",
    dataset_repo: str = "g-ronimo/masked_background_v3",
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    num_steps: int = 4_000,
    rank: int = 4,
    alpha: int = 4,
    validate_every: int = 200,
    save_every: int = 200,
    device = "cuda",
    dtype = torch.bfloat16,
    ):

    set_seed(42)
    pretrained_model_name_or_path = "black-forest-labs/FLUX.1-Fill-dev"
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
    [model.to(device, dtype=dtype) for model in [vae, text_encoder_one, text_encoder_two]]

    # Load Transformer
    print("Loading diffusion transformer")
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype=dtype
    )
    transformer.to(device, dtype=dtype)

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

    train_dataloader = get_maskeddataset_dataloader(dataset_repo, batch_size, resolution)

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

    # Setup Logging 
    wandb.init(
        project="FLUX-fill LoRA", 
    ).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

    # Try to fail early
    create_target_repo(target_repo)

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

    while global_step < num_steps:
        transformer.train()
            
        for batch in train_dataloader:
            if global_step > num_steps:
                break

            prompts = batch["prompts"]

            # 1 Image latent
            model_input = vae.encode(
                batch["images"].to(device).reshape(batch["images"].shape).to(transformer.dtype)
            ).latent_dist.sample()
            model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
            model_input = model_input.to(transformer.dtype)
        
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            
            # 2 Masked Image latent
            masked_image_latents = vae.encode(
                batch["images_masked"].to(device).reshape(batch["images"].shape).to(transformer.dtype)
            ).latent_dist.sample()
            
            masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
            
            # 3 Mask latent
            mask = batch["masks"]    
            mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
            mask = mask.view(
                model_input.shape[0], model_input.shape[2], vae_scale_factor, model_input.shape[3], vae_scale_factor
            )  # batch_size, height, 8, width, 8
            mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
            mask = mask.reshape(
                model_input.shape[0], vae_scale_factor * vae_scale_factor, model_input.shape[2], model_input.shape[3]
            )  
            
            latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                device,
                transformer.dtype,
            )
            
            bsz = model_input.shape[0]
            
            # TODO: Simplify this, we don't need the scheduler to sample a timestep!
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
            
            noise = torch.randn_like(model_input)
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
            
            # Concat 2+3
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
            
            # Concat 1+2+3
            transformer_input = torch.cat((packed_noisy_model_input, masked_image_latents), dim=2)    
            
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
                
            
            # Always log
            epoch = global_step / len(train_dataloader)
            print(f"step {global_step}, epoch {epoch:.2f}, loss {loss.detach().item()}, grad_norm: {grad_norm}")
            logs = dict(
                step = global_step,
                epoch = epoch,
                loss = loss.detach().item(), 
                lr = lr_scheduler.get_last_lr()[0], 
                grad_norm = grad_norm,
            )
            wandb.log(logs)

            # Validate 
            if global_step % validate_every == 0:
                images = run_validation(
                    pipeline=pipeline,
                    validation_prompt=validation_prompt,
                )
                wandb.log(dict(validation = wandb.Image(create_image_gallery(images))))
        
            # Save LoRA weights
            if global_step % save_every == 0 or global_step == num_steps:
                upload_adapter(transformer, target_repo, target_fn = f"adapter_step-{global_step}")

            global_step += 1

    wandb.finish()

    os.system('runpodctl remove pod $RUNPOD_POD_ID')

if __name__ == "__main__":
    Fire(main)
