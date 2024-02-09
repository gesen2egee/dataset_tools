#需先安裝 pip install numpy Pillow

import numpy as np
from PIL import Image
import argparse
import os
import re
        
class ImageAugmenterNP:

    def prune_background_tags(self, txt_path):
        background_tags = [
            r"simple.?background.?\s?",
            r"white.?background.?\s?",
            r"transparent.?background.?\s?",
            r"dark.?background.?\s?",
            r"black.?background.?\s?",
            r"grey.?background.?\s?",
            r"blue.?background.?\s?",
            r"purple.?background.?\s?",
            r"pink.?background.?\s?",
            r"yellow.?background.?\s?",
            r"orange.?background.?\s?",
            r"red.?background.?\s?",
            r"green.?background.?\s?",
            r"brown.?background.?\s?",
            r"aqua.?background.?\s?",
            r"beige.?background.?\s?",
            r"sepia.?background.?\s?",
            r"silver.?background.?\s?",
            r"light.?blue.?background.?\s?",
            r"light.?brown.?background.?\s?"
        ]
        try:
            with open(txt_path, 'r+', encoding='utf-8') as f:
                content = f.read()
                for tag in background_tags:
                    content = re.sub(tag, '', content, flags=re.IGNORECASE)
                f.seek(0)
                f.write(content)
                f.truncate()
        except FileNotFoundError:
            print(f"無法找到或開啟文本文件：{txt_path}")
            
    def calculate_edge_colors(self, image_np, threshold):
        edge_width = max(1, min(image_np.shape[0], image_np.shape[1]) // 20)  # 設定邊緣寬度
        # 上邊緣和下邊緣
        top_edge = image_np[:edge_width, :, :]
        bottom_edge = image_np[-edge_width:, :, :]
        # 左邊緣和右邊緣，進行轉置以匹配形狀
        left_edge = image_np[:, :edge_width, :].reshape(-1, image_np.shape[2])
        right_edge = image_np[:, -edge_width:, :].reshape(-1, image_np.shape[2])
        # 將左右邊緣的數據重塑後與上下邊緣數據形狀一致
        edges = np.concatenate([top_edge.reshape(-1, image_np.shape[2]), 
                                bottom_edge.reshape(-1, image_np.shape[2]),
                                left_edge, right_edge])
        # 使用重塑後的數據來計算顏色和計數
        colors, counts = np.unique(edges, axis=0, return_counts=True)
        simple_color = colors[counts.argmax()]
        simple_colorratio = counts.max() / counts.sum()

        return simple_color, simple_colorratio > threshold

    def process_images_from_folder(self, folder_path: str, mask_threshold: float, simple_background: bool = False, prune_background_tag: bool = False):
        """
        讀取腳本所在第一級子資料夾內所有圖片(排除mask資料夾)，
        將圖片中的透明、純黑(0, 0, 0)和純白(255, 255, 255)像素設為蒙版，
        其他顏色像素將被設定為非蒙版，並將結果保存為png到每個子資料夾的mask子資料夾中。
        
        :param -t (小数，預設0.0) : mask_threshold-如果蒙版像素所佔比例未超過設定的門檻值(預設為0)，則不保存PNG圖像。
        :param -s : simple_background-是否統計圖片邊緣的顏色，判斷是否單色背景並加入蒙版 (預設門檻 0.5)。
        :param -p : prune_background_tag-如果蒙版像素所佔全圖比例較高(預設門檻 0.3)，自動刪除同名txt中的background_tags。
        """    
        for dir_name in next(os.walk(folder_path))[1]:
            current_folder = os.path.join(folder_path, dir_name)
            target_folder = os.path.join(current_folder, 'mask')
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                print(f"建立目標資料夾：{target_folder}")

            for filename in os.listdir(current_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(current_folder, filename)
                    if 'mask' in image_path:
                        continue
                    try:
                        with Image.open(image_path) as image:
                            if image.mode != 'RGBA':
                                image = image.convert('RGBA')
                                
                            # 使用Numpy進行圖像處理
                            image_np = np.array(image)
                            is_simple = False
                            if simple_background:
                                # 檢測單色背景
                                simple_color, is_simple = self.calculate_edge_colors(image_np,0.5)                            
                            black_bg_np = np.zeros_like(image_np)
                            black_bg_np[:, :, 3] = 255  # 確保背景是不透明的
                            merged_image_np = np.maximum(black_bg_np, image_np)  # 將圖像合併到純黑背景上

                            # 條件過濾與設置蒙版
                            mask_white = (merged_image_np[:, :, :3] == [255, 255, 255]).all(axis=2)
                            mask_black = (merged_image_np[:, :, :3] == [0, 0, 0]).all(axis=2)
                            mask = mask_white | mask_black
                            if is_simple:
                                simple_mask = np.all(image_np[:, :, :3] == simple_color[:3], axis=2)
                                mask |= simple_mask                                
                            merged_image_np[~mask] = [255, 255, 255, 255]  # 非蒙版設為白色
                            merged_image_np[mask] = [0, 0, 0, 255]  # 蒙版設為黑色

                            # 計算蒙版比例
                            black_pixels_ratio = np.mean(mask)

                            if black_pixels_ratio > mask_threshold:
                                image_result = Image.fromarray(merged_image_np).convert('RGB')
                                base_filename = os.path.splitext(filename)[0]
                                save_path = os.path.join(target_folder, f"{base_filename}.png")
                                image_result.save(save_path, "PNG")
                                print(f"圖片 {filename} 已處理並保存於 {save_path}")
                                if prune_background_tag and black_pixels_ratio > 0.3:
                                    txt_path = os.path.splitext(image_path)[0] + '.txt'
                                    self.prune_background_tags(txt_path)

                    except FileNotFoundError as e:
                        print(f"無法找到文件：{image_path}")
                        continue

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image augmentation tool with options for mask generation and single color background detection.')
    parser.add_argument('-t', '--mask_threshold', type=float, default=0.0, help='Threshold for color difference to decide if a pixel belongs to the subject.')
    parser.add_argument('-s', '--simple_background', action='store_true', help='Enable detection of simple (single-color) backgrounds with a subject.')
    parser.add_argument('-p', '--prune_background_tag', action='store_true', help='Enable auto pruning of background tags in the corresponding text file.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    augmenter = ImageAugmenterNP()
    augmenter.process_images_from_folder('.', args.mask_threshold, args.simple_background, args.prune_background_tag)