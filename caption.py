import subprocess
import os
import platform
import argparse
import sys
def install_and_import(package):
    try:
        import (package)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        import (package)

install_and_import("requests")

def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise一个 HTTPError如果 HTTP 请求返回了一个不成功的状态码
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename} from {url}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename} from {url}: {e}")
        sys.exit(1)

def run_setup_script():
    setup_url = "https://raw.githubusercontent.com/gesen2egee/dataset_tools/main/setup.py"
    setup_filename = "setup.py"
    main_script_filename = "main_script.py"
    
    if not os.path.exists(setup_filename) or args.upgrade:
        download_file(setup_url, setup_filename)
    if not os.path.exists(main_script_filename) or args.upgrade:
        subprocess.run([sys.executable, setup_filename], check=True)

def run_main_script_in_venv(args):
    main_script_url = "https://raw.githubusercontent.com/gesen2egee/dataset_tools/main/main_script.py"
    main_script_filename = "main_script.py"

    if not os.path.exists(main_script_filename) or args.upgrade:
        download_file(main_script_url, main_script_filename)

    if platform.system() == 'Windows':
        activate_script = os.path.join('venv', 'Scripts', 'activate.bat')
    else:
        activate_script = os.path.join('venv', 'bin', 'activate')

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

    if os.name == 'nt':
        command = f"{activate_script} && python {main_script_filename} " + subprocess.list2cmdline(command_args)
    else:
        command = f"source {activate_script} && python {main_script_filename} " + subprocess.list2cmdline(command_args)

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
    run_main_script_in_venv(args)
