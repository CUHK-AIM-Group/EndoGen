# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import time
import argparse
from tqdm import tqdm
import numpy as np
import moviepy.editor as mp
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate


# convert to gif
def convert_to_gif(video_path, gif_path):
    clip = mp.VideoFileClip(video_path)
    clip.write_gif(gif_path)
    print(f"gif is saved to {gif_path}")

import cv2
def convert_4x4_to_16frame_video(image_path, output_video_path, frame_size=(128, 128), fps=1):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the size of each cell in the grid
    cell_height = height // 4
    cell_width = width // 4

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Extract each cell and write it as a frame in the video
    for i in range(4):
        for j in range(4):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            resized_cell = cv2.resize(cell, frame_size)
            video_writer.write(resized_cell)

    # Release the VideoWriter object
    video_writer.release()

def convert_4x4_to_16frame_folder(image_path, output_frame_folder, frame_size=(128, 128), fps=1):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the size of each cell in the grid
    cell_height = height // 4
    cell_width = width // 4

    # create folder
    if not os.path.exists(output_frame_folder):
        os.makedirs(output_frame_folder)

    # Extract each cell and write it as a frame in the video
    # id is row by row, 000000, 000001, 000002, 000003, 000004, 000005, 000006, 000007, 000008, 000009
    start_id = 0
    for i in range(4):
        for j in range(4):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            resized_cell = cv2.resize(cell, frame_size)
            frame_path = os.path.join(output_frame_folder, f"{start_id:06d}.png")
            cv2.imwrite(frame_path, resized_cell)
            start_id += 1

    print(f"16 frame images are saved to {output_frame_folder}")

def main(args):
    # Setup PyTorch:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 

    if args.num_classes == 8:
        mapping = {0: 'barretts', 1: 'cancer', 2: 'esophagitis', 3: 'gastric-antral-vascular-ectasia', 4: 'gastric-banding-perforated', 5: 'polyps', 6: 'ulcer', 7: 'varices'}
    elif args.num_classes == 3:
        mapping = {0: 'Dissection', 1: 'Knot_Tying', 2: 'Needle_Driving'}
    else:
        raise ValueError(f"num classes {args.num_classes} is not supported")
    class_labels = [i for i in range(args.num_classes)] # generete 1 video for each class
    n_generates = len(class_labels) # number of videos to generate, one for each class
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, latent_size ** 2,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"generation takes about {sampling_time:.2f} seconds.")
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_16frame_folder = os.path.join(save_folder, f"16frame_folder")
    if not os.path.exists(save_16frame_folder):
        os.makedirs(save_16frame_folder)
    assert samples.shape[0] == n_generates, f"sample shape is {samples.shape}"
    if 'cfg' not in args.name:
        args.name = args.name + "cfg" + str(args.cfg_scale)
    for cls_id in range(samples.shape[0]):
        class_name = mapping[cls_id]
        save_image(samples[cls_id], f"{save_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.png", normalize=True, value_range=(-1, 1))
        print(f"image is saved to {class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.png")
        convert_4x4_to_16frame_folder(f"{save_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.png",
                                    f"{save_16frame_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}",
                                    frame_size = (args.image_size//4, args.image_size//4))
        convert_4x4_to_16frame_video(f"{save_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.png", 
                                     f"{save_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.mp4",
                                     frame_size = (args.image_size//4, args.image_size//4))
        print(f"video is saved to {save_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.mp4")
        convert_to_gif(f"{save_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.mp4", 
                       f"{save_folder}/{class_name}_sample_seed{seed}_{args.name}_{args.gpt_type}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512, 1024], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--save-folder", type=str, default="sampled_images", help="folder to save images")
    parser.add_argument("--name", type=str, default="sample", help="name of the experiment")
    args = parser.parse_args()
    main(args)