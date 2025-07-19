import os, shutil, itertools, random, torch
import numpy as np
from transformers import (
    PretrainedConfig, 
    CLIPTokenizer,
    CLIPTextModel, 
    T5TokenizerFast,
    T5EncoderModel,
)
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from torchvision.transforms.functional import crop
from datasets import load_dataset
from PIL import Image
import math
import random

def load_images_and_masks_from_folder(folder):
    image_mask_pairs = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpeg', '.jpg', '.png')) and '_mask' not in filename:
            image_path = os.path.join(folder, filename)
            mask_filename = filename.split('.')[0] + '_mask.png'
            mask_path = os.path.join(folder, mask_filename)

            if os.path.exists(mask_path):
                try:
                    img = Image.open(image_path)
                    mask = Image.open(mask_path)
                    if img is not None and mask is not None:
                        image_mask_pairs.append((img, mask))
                except IOError as e:
                    print(f"Error opening files {filename} or {mask_filename}: {e}")
            else:
                print(f"Mask file not found for {filename}")

    return image_mask_pairs

def create_image_gallery(images, cell_size=300, background_color=(255, 255, 255)):
    """
    Create a square gallery from a list of PIL images.
    
    Args:
        images: List of PIL Image objects
        cell_size: Size of each cell in the grid (default: 300)
        background_color: Background color as RGB tuple (default: white)
    
    Returns:
        PIL Image object containing the gallery
    """
    if not images:
        raise ValueError("Image list cannot be empty")
    
    # Calculate grid dimensions (square grid)
    grid_size = math.ceil(math.sqrt(len(images)))
    
    # Create the output image
    gallery_width = grid_size * cell_size
    gallery_height = grid_size * cell_size
    gallery = Image.new('RGB', (gallery_width, gallery_height), background_color)
    
    # Process each image
    for i, img in enumerate(images):
        # Calculate grid position
        row = i // grid_size
        col = i % grid_size
        
        # Calculate position in the gallery
        x = col * cell_size
        y = row * cell_size
        
        # Resize image to fit in cell while preserving aspect ratio
        img_resized = resize_with_aspect_ratio(img, cell_size)
        
        # Center the image within the cell
        img_width, img_height = img_resized.size
        center_x = x + (cell_size - img_width) // 2
        center_y = y + (cell_size - img_height) // 2
        
        # Paste the image onto the gallery
        gallery.paste(img_resized, (center_x, center_y))
    
    return gallery

def resize_with_aspect_ratio(img, max_size):
    """
    Resize an image to fit within max_size while preserving aspect ratio.
    
    Args:
        img: PIL Image object
        max_size: Maximum width or height
    
    Returns:
        Resized PIL Image object
    """
    original_width, original_height = img.size
    
    # Calculate scaling factor
    scale = min(max_size / original_width, max_size / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def load_text_encoders(pretrained_model_name_or_path):
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder", 
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="text_encoder_2", 
    )

    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two


def remove_cache_and_checkpoints(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.cache' in dirnames:
            shutil.rmtree(os.path.join(dirpath, '.cache'))
        if '.ipynb_checkpoints' in dirnames:
            shutil.rmtree(os.path.join(dirpath, '.ipynb_checkpoints'))

def get_maskeddataset_dataloader(hf_repo, batch_size, resolution):
    ds = MaskedDataset(
        hf_repo, 
        resolution = resolution,
        resizeTo = resolution
    )

    train_dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = lambda examples: collate_fn(examples),
        # num_workers=args.dataloader_num_workers,
    )

    return train_dataloader

class MaskedDataset(Dataset):
    def __init__(
        self,
        hf_dataset_name,
        resolution,
        resizeTo=1024,
        hf_dataset_split = "train",
    ):
        # Load dataset 
        self.hf_dataset = load_dataset(hf_dataset_name)[hf_dataset_split]

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resizeTo, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resizeTo),
            ]
        )
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        image = self.hf_dataset[index]["image"]
        image = image.convert("RGB") if not image.mode == "RGB" else image
        image = self.image_transforms(image)        

        # resize mask to match PIL
        mask = self.hf_dataset[index]["mask"].convert("L")
        mask = mask.resize(image.size, Image.NEAREST)

        prompt = self.hf_dataset[index]["prompt"]
        
        return dict(
            image = image,
            mask = mask,
            prompt = prompt
        )


def collate_fn(examples):
    # PIL->normalized tensor
    pil_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # text
    prompts = [example["prompt"] for example in examples]
    # tensors
    images, images_masked, masks  = [], [], []

    for example in examples:
        mask, image_masked = prepare_mask_and_masked_image(example["image"], example["mask"])

        masks.append(mask)
        images_masked.append(image_masked)
        images.append(pil_to_tensor(example["image"]))

    return dict(
        images = torch.stack(images).to(memory_format=torch.contiguous_format).float(), 
        images_masked = torch.stack(images_masked),
        masks = torch.stack(masks), 
        prompts = prompts, 
    )


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)
    
    return mask, masked_image

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length):
    device = text_encoders[0].device

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def prepare_mask_latents(
        mask,
        masked_image_latents,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        vae_scale_factor,
        vae_shift_factor
     ):
        """ Prepare mask latents """
        # 1. calculate the height and width of the latents
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

  

        masked_image_latents = (masked_image_latents - vae_shift_factor) * vae_scale_factor #self.vae.config.scaling_factor
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        # 3. duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        batch_size = batch_size * num_images_per_prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # 4. pack the masked_image_latents
        # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4
        masked_image_latents = FluxFillPipeline._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        # 5.resize mask to latents shape we we concatenate the mask to the latents
        mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(
            batch_size, height, vae_scale_factor, width, vae_scale_factor
        )  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
            batch_size, vae_scale_factor * vae_scale_factor, height, width
        )  # batch_size, 8*8, height, width

        # 6. pack the mask:
        # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
        mask = FluxFillPipeline._pack_latents(
            mask,
            batch_size,
            vae_scale_factor * vae_scale_factor,
            height,
            width,
        )
        mask = mask.to(device=device, dtype=dtype)

        return mask, masked_image_latents

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def get_sigmas(timesteps, noise_scheduler, n_dim=4, device="cuda", dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma



