import subprocess
import os
import platform
import argparse
import requests
import sys

def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename} from {url}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename} from {url}: {e}")
        sys.exit(1)

def run_setup_script():
    setup_url = "https://raw.githubusercontent.com/gesen2egee/dataset_tools/main/setup.py"
    setup_filename = "setup.py"
    
    if not os.path.exists(setup_filename):
        download_file(setup_url, setup_filename)
    
    subprocess.run([sys.executable, setup_filename], check=True)

def run_main_script_in_venv(args):
    main_script_url = "https://raw.githubusercontent.com/gesen2egee/dataset_tools/main/main_script.py"
    main_script_filename = "main_script.py"

    if not os.path.exists(main_script_filename):
        download_file(main_script_url, main_script_filename)

    if platform.system() == 'Windows':
        activate_script = os.path.join('venv', 'Scripts', 'activate.bat')
    else:
        activate_script = os.path.join('venv', 'bin', 'activate')

    command_args = " ".join([
        "--folder_name" if args.folder_name else "",
        "--drop_chartag" if args.drop_chartag else "",
        "--not_char" if args.not_char else "",
        "--use_norm" if args.use_norm else "",
        f"--continue_caption {args.continue_caption}" if args.continue_caption else "",
        args.directory
    ]).strip()

    command = f"{activate_script} && python {main_script_filename} {command_args}" if os.name == 'nt' else f"source {activate_script} && python {main_script_filename} {command_args}"
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在虛擬環境中運行主腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--drop_chartag", action="store_true", help="自動刪除角色特徵標籤")
    parser.add_argument("--not_char", action="store_true", help="目錄名不是角色")
    parser.add_argument("--use_norm", action="store_true", help="忽略clip文字向量長度，標會較短")
    parser.add_argument("--continue_caption", type=int, default=0, help="忽略n天內打的標")    
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()

    run_setup_script()
    run_main_script_in_venv(args)
