import numpy as np
from PIL import Image
import argparse
import os

class ImageAugmenterNP:
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

    def process_images_from_folder(self, folder_path: str, threshold: float, simple_background: bool = False):
        """
        讀取腳本所在資料夾及其第一級子資料夾內所有圖片，
        將圖片中的透明、純黑(0, 0, 0)和純白(255, 255, 255)像素設為蒙版，
        其他顏色像素將被設定為非蒙版，並將結果保存為png到每個子資料夾的mask子資料夾中。
        
        如果蒙版所佔比例未超過設定的門檻值(預設為0)，則不保存PNG圖像。
        :param threshold: 黑色像素所佔全圖比例的門檻值。
        :param simple_background: 是否統計圖片邊緣的顏色，判斷是否單色背景並加入蒙版 (預設門檻 0.5)。
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
    parser = argparse.ArgumentParser(description='Image augmentation tool with options for mask generation and single color background detection.')
    parser.add_argument('-t', '--threshold', type=float, default=0.0, help='Threshold for color difference to decide if a pixel belongs to the subject.')
    parser.add_argument('-s', '--simple_background', action='store_true', help='Enable detection of simple (single-color) backgrounds with a subject.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    augmenter = ImageAugmenterNP()
    augmenter.process_images_from_folder('.', args.threshold, args.simple_background)