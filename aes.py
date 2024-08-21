import subprocess
import sys
from pathlib import Path
import torch
import shutil
import re
from tqdm import tqdm
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image
from imgutils.metrics import get_aesthetic_score, anime_dbaesthetic
from imgutils.validate import anime_completeness
from imgutils.metrics import laplacian_score

# 設定模型和預處理器
model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = model.to(torch.bfloat16).cuda()

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .cuda()
    )
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()
        #score = get_aesthetic_score(image) * (8 - 4) + 4
        #score = (db_score+aes_score) / 2
    return score

def get_aesthetic_tag(score):
    if score >= 5.5:
        return "aesthetic"
    elif score >= 4.5:
        return "good"
    else:
        return "rough"

def move_images_to_folders(src_folder):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.BMP"]
    for ext in extensions:
        for image_path in src_folder.glob(ext):
            tag, score = anime_completeness(image_path)
            if tag == "polished":
                score = process_image(image_path)
                tag = get_aesthetic_tag(score)
            target_folder = src_folder / tag
            target_folder.mkdir(exist_ok=True)
            target_image_path = target_folder / image_path.name
            shutil.move(str(image_path), str(target_image_path))
            print(f"Moved {image_path} to {target_image_path}")

def main():
    base_path = Path.cwd()
    cat = ["aesthetic",  "good", "rough", "monochrome"]
    subfolders = [f for f in base_path.iterdir() if f.is_dir() and f not in cat]

    for folder in tqdm(subfolders, desc="Processing folders"):
        move_images_to_folders(folder)

if __name__ == "__main__":
    main()
