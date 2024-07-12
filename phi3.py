import os
import sys
import shutil
from io import BytesIO
from pathlib import Path
from glob import glob
from tqdm import tqdm
import re
import json
import traceback
import fnmatch
import torch
import torch.nn.functional as F

from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "Desm0nt/Phi-3-HornyVision-128k-instruct"
kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2', cache_dir="./cache") # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir="./cache")

system_prompt = '<|system|>\n'
user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"

custom_system_prompt = "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, attire, actions, pose, expressions, accessories, makeup, composition type, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 1-5 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. DO NOT REPEAT YOURSELF.<|end|>\n"

custom_prompt = "Make a JSON object that describe this image, following this format \"{ \"description\": \"Describe the image within 150 words with this format \"This image include {main object}, {setting or actions with pose}, {describe the background}, {setting, artistic style and composition type}, and any other for the image.\"\", \"tags\": \"Describe the image within 15 tags separated by commas.\"}\""
prompt = f"{system_prompt}{custom_system_prompt}{prompt_suffix}{user_prompt}<|image_1|>\n{custom_prompt}{prompt_suffix}{assistant_prompt}"

def process_image(image_path, args):
    """
    縮小圖像使其最大邊不超過 max_size，返回縮小後的圖像數據
    """
    def resize_image(image_path, max_size=448):
        image = Image.open(image_path)
        if max(image.width, image.height) > max_size:
            if image.width > image.height:
                new_width = max_size
                new_height = int(max_size * image.height / image.width)
            else:
                new_height = max_size
                new_width = int(max_size * image.width / image.height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image

    image = resize_image(image_path)

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1256,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.05,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    try:
        response = json.loads(response)
    except Exception as e:
        response = dict({"description": None, "tags": None})

    return response['description'], response['tags']

def find_and_process_images(directory, args):
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    all_final_scores = []
    for root, dirs, files in os.walk(directory):
        folder_chartag = {}
        image_paths = []
        image_infos_list = []
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

        chartag_from_folder = ''
        parent_folder = Path(directory).name
        if args.folder_name and '_' in parent_folder and parent_folder.split('_')[0].isdigit():
            chartag_from_folder = ' '.join(char.strip() for char in parent_folder.split('_')[1:])

        if chartag_from_folder != '':
            chartag_from_folder += ', '

        trigger_word = args.trigger_word.strip()
        if trigger_word != '':
            trigger_word += ', '

        for image_path in tqdm(image_paths, desc=f"處理圖片 {root}"):
            try:
                description, tags = process_image(image_path, args)

                if description != None and tags != None:
                    content = f"{trigger_word}{chartag_from_folder}___{tags}___ {description}\n"
                    content += f"{trigger_word}{chartag_from_folder}{description}\n"
                    content += f"{trigger_word}{chartag_from_folder}___{tags}___"
                    content = content.replace(',___', '___').lower()

                    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')

                    if tag_file_path.exists() == False or args.override == True:
                        with open(tag_file_path, 'w', encoding='utf-8') as file:
                            file.write(content)
                            file.close()
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--trigger_word", type="str", default="", help="使用目錄名當作角色名")
    parser.add_argument("--override", action="store_true", help="覆蓋既有標籤檔案")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()

    find_and_process_images(args.directory, args)
