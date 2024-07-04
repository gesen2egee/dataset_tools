import subprocess
import os
import platform
import argparse
import sys
import shutil

def download_file(url, filename):
    import requests
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise一个 HTTPError如果 HTTP 请求返回了一个不成功的状态码
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename} from {url}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename} from {url}: {e}")
        sys.exit(1)

def run_setup_script(venv_name="venv"):
    # 检查并创建虚拟环境
    if not os.path.exists(venv_name):
        print(f"Creating virtual environment: {venv_name}")
        subprocess.run([sys.executable, '-m', 'venv', venv_name])
    else:
        print(f"Virtual environment '{venv_name}' already exists.")

    if platform.system() == 'Windows':
        activate_script = os.path.join('venv', 'Scripts', 'activate.bat')
    else:
        activate_script = os.path.join('venv', 'bin', 'activate')

    flash_attn = "flash_attn==2.5.9.post1"
    common_command = f"pip install -r requirements.txt && pip install {flash_attn} && pip cache purge"

    print("Installing all modules")
    if platform.system() == 'Windows':
        command = f"{activate_script} && {common_command}"
    else:
        command = f". {activate_script} && {common_command}"

    subprocess.run(command, check=True)

def setup_longclip():
    # 检查 checkpoints 目录是否存在
    if not os.path.exists('./checkpoints'):
        # 克隆仓库
        if not os.path.exists('Long-CLIP'):
            subprocess.run(['git', 'clone', 'https://github.com/beichenzbc/Long-CLIP'])

        # 移动文件
        for filename in os.listdir('Long-CLIP'):
            shutil.move(os.path.join('Long-CLIP', filename), '.')

        # 删除克隆的目录
        shutil.rmtree('Long-CLIP')

    # 下载权重文件
    weight_url = 'https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14/resolve/main/Long-ViT-L-14-GmP-ft-state_dict.pt?download=true'
    weight_path = './checkpoints/Long-ViT-L-14-GmP-ft-state_dict.pt'
    download_file(weight_url, weight_path)

def run_main_script_in_venv(args):
    main_script_url = "https://raw.githubusercontent.com/gesen2egee/dataset_tools/main/main_script.py"
    main_script_filename = "main_script.py"

    if not os.path.exists(main_script_filename) or args.upgrade:
        download_file(main_script_url, main_script_filename)

    command_args = [
        args.directory,
        "--folder_name" if args.folder_name else "",
        "--drop_chartag" if args.drop_chartag else "",
        "--drop_colortag" if args.drop_colortag else "",
        "--clothtag" if args.clothtag else "",
        "--peopletag" if args.peopletag else "",
        "--not_char" if args.not_char else "",
        "--debiased" if args.debiased else "",
        "--rawdata" if args.rawdata else "",
        f"--custom_keeptag=\"{args.custom_keeptag}\"" if args.custom_keeptag else "",
        f"--continue_caption {args.continue_caption}" if args.continue_caption else ""
    ]

    # 过滤掉空字符串
    command_args = [arg for arg in command_args if arg]
    if platform.system() == 'Windows':
        command = f"{activate_script} && python {main_script_filename} " + subprocess.list2cmdline(command_args)
    else:
        command = f". {activate_script} && python {main_script_filename} " + subprocess.list2cmdline(command_args)

    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在虛擬環境中運行主腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--drop_chartag", action="store_true", help="自動刪除角色特徵標籤")
    parser.add_argument("--drop_colortag", action="store_true", help="自動刪除顏色特徵標籤")
    parser.add_argument("--clothtag", action="store_true", help="自動處理服裝標籤")
    parser.add_argument("--not_char", action="store_true", help="目錄名不是角色")
    parser.add_argument("--debiased", action="store_true", help="設置clip score上限減少florence偏差")
    parser.add_argument("--peopletag", action="store_true", help="前置多人標籤(大多nsfw)")
    parser.add_argument("--rawdata", action="store_true", help="大資料集")
    parser.add_argument("--custom_keeptag", type=str, default=None, help="自定義自動留標")
    parser.add_argument("--continue_caption", type=int, default=0, help="忽略n天內打的標")
    parser.add_argument("--upgrade", action="store_true", help="升級腳本")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()

    run_setup_script()
    setup_longclip()
    run_main_script_in_venv(args)
