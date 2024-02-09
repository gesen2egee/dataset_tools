import numpy as np
from PIL import Image
import argparse
import os

class ImageAugmenterNP:
    def process_images_from_folder(self, folder_path: str, threshold: float):
        """
        讀取腳本所在資料夾及其第一級子資料夾內所有圖片，
        將圖片中的透明、純黑(0, 0, 0)和純白(255, 255, 255)像素設為蒙版，
        其他顏色像素將被設定為非蒙版，並將結果保存為png到每個子資料夾的mask子資料夾中。
        
        如果蒙版所佔比例未超過設定的門檻值(預設為0)，則不保存PNG圖像。
        :param threshold: 黑色像素所佔全圖比例的門檻值。
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
                            black_bg_np = np.zeros_like(image_np)
                            black_bg_np[:, :, 3] = 255  # 確保背景是不透明的
                            merged_image_np = np.maximum(black_bg_np, image_np)  # 將圖像合併到純黑背景上

                            # 條件過濾與設置蒙版
                            mask_white = (merged_image_np[:, :, :3] == [255, 255, 255]).all(axis=2)
                            mask_black = (merged_image_np[:, :, :3] == [0, 0, 0]).all(axis=2)
                            mask = (mask_white | mask_black)

                            merged_image_np[~mask] = [255, 255, 255, 255]  # 非蒙版設為白色
                            merged_image_np[mask] = [0, 0, 0, 255]  # 蒙版設為黑色

                            # 計算蒙版比例
                            black_pixels_ratio = np.mean(mask)

                            if black_pixels_ratio > threshold:
                                image_result = Image.fromarray(merged_image_np).convert('RGB')
                                base_filename = os.path.splitext(filename)[0]
                                save_path = os.path.join(target_folder, f"{base_filename}.png")
                                image_result.save(save_path, "PNG")
                                print(f"圖片 {filename} 已處理並保存於 {save_path}")

                    except FileNotFoundError as e:
                        print(f"無法找到文件：{image_path}")
                        continue

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--threshold', type=float, default=0.0, help='Threshold for mask pixel ratio to decide if saving a mask png.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    augmenter = ImageAugmenterNP()
    augmenter.process_images_from_folder('.', args.threshold)