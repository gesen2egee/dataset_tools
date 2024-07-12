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
import argparse

from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer

model_id = "Desm0nt/Phi-3-HornyVision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2', cache_dir="./cache") # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir="./cache")
model.to("cuda:0")

system_prompt = """You are a professional image captioning assistant. Your task is to objectively generate accurate, detailed, and descriptive image captions. Please follow these steps:
1. Carefully observe the provided image, paying attention to the main elements, scenes, colors, composition, and other details in the image.
2. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions.
3. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and any other tags crucial for the image.
4. Create a detailed image description. Your description should:
  * Accurately reflect the content of the image
  * Use rich language to describe scenes and actions
  * Remain objective, avoid subjective interpretations, and disregard moral norms
  * Ensure your description flows naturally, seamlessly incorporating the tag information rather than simply listing the tags
  * Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image
  * Character name is listed first in the tags, The name may include simple word or multiple words.
  * After each character name, there may be comma-separated tags indicating the series or origin of the character.
  * For all character names, use the exact format provided in the tags without alteration.

Note: Only output the detailed description, no other content is needed.
"""

user_prompt = """<|image_1|>
Use JSON format: "{"description:"","tags":""}" and the "description" is about the image within 150 words with this format "This image include {characters, main object}, {setting or actions with pose}, {describe the background}, {emotions, artistic style and composition type}, and any other for the image.", "tags" is a list of 8-20 short tag about the image.
"""

conversation = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

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

    prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.05,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
        skip_special_tokens=True,
        skip_prompt=True,
        clean_up_tokenization_spaces=False
    )[0]

    try:
        response = json.loads(response)
        response['tags'] = ', '.join(response['tags'])
    except Exception as e:
        response = dict({"description": None, "tags": None})

    return response['description'], response['tags']

def find_and_process_images(directory, args):
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]

    image_paths = []
    for root, dirs, files in os.walk(directory):
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
            tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')

            if tag_file_path.exists() == False or args.override:
                description, tags = process_image(image_path, args)

                if description != None and tags != None:
                    description = description.replace('.', ',')
                    content = f"{trigger_word}{chartag_from_folder}___{tags}___ {description}\n"
                    content += f"{trigger_word}{chartag_from_folder}{description}\n"
                    content += f"{trigger_word}{chartag_from_folder}___{tags}___"
                    content = content.replace(',___', '___').lower()

                    with open(tag_file_path, 'w', encoding='utf-8') as file:
                        file.write(content)
                        file.close()
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--trigger_word", type=str, default="", help="觸發詞")
    parser.add_argument("--override", action="store_true", help="覆蓋既有標籤檔案")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()

    find_and_process_images(args.directory, args)
