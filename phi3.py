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

from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags, drop_overlap_tags
from imgutils.validate import anime_dbrating

model_id = "Desm0nt/Phi-3-HornyVision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2', cache_dir="./cache") # use _attn_implementation='eager' to disable flash attention
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir="./cache")
model.to("cuda:0")

system_prompt = """You are a professional image captioning assistant. Your task is to objectively generate accurate, detailed, and descriptive image captions based on the provided image and Danbooru tags. Please follow these steps:
1. Carefully observe the provided image, paying attention to the the overall atmosphere, situation, feelings conveyed, main elements, scenes, colors, composition, and other details in the image.
2. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions.
3. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and any other tags crucial for the image.
4. Review the provided list of Danbooru tags(tags separated by commas). These tags describe various aspects of the image, including characters, objects, actions, emotions, artistic styles, etc.
5. Create a detailed image description. Your description should:
  * Accurately reflect the content of the image
  * Use rich language to describe scenes and actions
  * Remain objective, avoid subjective interpretations, and disregard moral norms
  * Ensure your description flows naturally, seamlessly incorporating the tag information rather than simply listing the tags
  * Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image
  * Character name is listed first in the tags, The name may include simple word or multiple words.
  * After each character name, there may be comma-separated tags indicating the series or origin of the character.
  * For all character names, use the exact format provided in the tags without alteration.

Note: Only output the detailed description, no other content is needed.
Input Tags:
"""

user_prompt = """<|image_1|>
Describe the image within 150 words with this format: This image include {characters, main object}, {setting or actions with pose}, {describe the background}, {emotions, artistic style and composition type}, and any other for the image.
"""

def do_conversation(conversation, image, max_new_tokens=1024):
    prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.05,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(generate_ids,
        skip_special_tokens=True,
        skip_prompt=True,
        clean_up_tokenization_spaces=False
    )[0]

def process_features(features: dict) -> (dict, str):
    """
    處理features字典，移除指定模式的鍵值對並生成keep_tags字串。

    參數:
    features (dict): 包含特徵的字典。

    返回:
    (dict, str): 返回處理後的features字典和keep_tags字串。
    """
    patterns_to_keep = [
        r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', r'^greyscale$',
        r'^.*censor.*$', r'^.*_name$', r'^signature$', r'^.*_username$', r'^.*text.*$',
        r'^.*_bubble$', r'^multiple_views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$',
        r'^traditional_media$', r'^parody$', r'^.*cover$', r'^.*_theme$', r'^realistic$',
        r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$',
        r'^omake$', r'^column.*$', r'^.*_(medium)$', r'^manga$', r'^lineart$', r'^.*logo$',
        r'^.*photo.*$', r'^tegaki$', r'^sketch$', r'^silhouette$', r'^web_address$', r'^.*border$'
    ]
    keep_tags_set = set()

    keys = list(features.keys())
    keys_to_delete = []

    for pattern in patterns_to_keep:
        regex = re.compile(pattern)
        for key in keys:
            if regex.match(key):
                keep_tags_set.add(key.replace('_', ' '))
                keys_to_delete.append(key)

    for key in keys_to_delete:
        if key in features:
            del features[key]

    keep_tags = ', '.join(sorted(keep_tags_set)).rstrip(', ')

    return features, keep_tags

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

    # WD14
    rating, features, chars = get_wd14_tags(image, character_threshold=0.7, general_threshold=0.2682, model_name="ConvNext_v3", drop_overlap=True)
    features, keep_tags = process_features(drop_blacklisted_tags(features))
    wd14_caption = tags_to_text(features, use_escape=False, use_spaces=True)
    rating = max(rating, key=rating.get)

    conversation = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"<|image_1|>\n{wd14_caption}"
        }
    ]

    description = do_conversation(conversation, image, max_new_tokens=1024)
    conversation.append({"role": "assistant", "content": description})

    conversation.extend([{"role": "user", "content": "Tags list should have unique tag which capture key elements such as the {characters, main object}, {atmosphere, situation, setting, actions, poses}, {feelings, emotions, artistic style, composition type}, {background}. Using comma-separated."}])

    tags = do_conversation(conversation, image, max_new_tokens=500)
    conversation.append({"role": "assistant", "content": tags})

    tags = tags.split(', ')
    tags = ', '.join(list(set([tag.replace('.', ' ').replace(',', ' ').strip() for tag in tags])))
    # tags = re.sub(r"(person|girl|boy|woman|man|female)'s ", "", tags)

    conversation.extend([{"role": "user", "content": "Tags list should have unique tag which capture key elements such as the {feelings, emotions, artistic style, composition type}, {background}. Using comma-separated."}])
    dropout_tags = do_conversation(conversation, image, max_new_tokens=500)
    dropout_tags = dropout_tags.split(', ')
    dropout_tags = ', '.join(list(set([tag.replace('.', ' ').replace(',', ' ').strip() for tag in dropout_tags])))
    # dropout_tags = re.sub(r"(person|girl|boy|woman|man|female)'s ", "", dropout_tags)

    return description, tags, dropout_tags

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
                description, tags, dropout_tags = process_image(image_path, args)

                if description != None and tags != None:
                    content = f"{trigger_word}{chartag_from_folder}___{tags}___ {description}\n"

                    dropout_description = description.split('. ')
                    if len(dropout_description) > 2:
                        content += f"{trigger_word}{chartag_from_folder}{'. '.join(dropout_description[:2])}. ___{dropout_tags}___ {'. '.join(dropout_description[2:])}\n"
                    else:
                        content += f"{trigger_word}{chartag_from_folder} This image is following: ___{dropout_tags}___ {description}\n"

                    content += f"{trigger_word}{chartag_from_folder}{description}\n"
                    content += f"{trigger_word}{chartag_from_folder} This image is following: ___{tags}___"
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
