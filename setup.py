import subprocess
import os
import sys
import shutil
import platform
try:
    import requests
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    import requests
    
def install_packages(venv_name='venv'):
    def is_package_installed(package_name, activate_command):
        check_command = f"{activate_command}python -m pip show {package_name}"
        result = subprocess.run(check_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0

    # 检查并创建虚拟环境
    if not os.path.exists(venv_name):
        print(f"Creating virtual environment: {venv_name}")
        subprocess.run([sys.executable, '-m', 'venv', venv_name])
    else:
        print(f"Virtual environment '{venv_name}' already exists.")
    
    # 确定激活脚本路径
    if os.name == 'nt':
        activate_script = os.path.join(venv_name, 'Scripts', 'activate.bat')
    else:
        activate_script = os.path.join(venv_name, 'bin', 'activate')
    
    # 激活虚拟环境并安装依赖包
    activate_command = f"{activate_script} && " if os.name == 'nt' else f"source {activate_script} && "
    
    # 通用依赖包
    common_packages = [
        'argparse',
        'tqdm',
        'Pillow',
        'pathlib',
        'datetime',
        'transformers',
        'inflect',
        'onnxruntime-gpu==1.17.0',
        'ftfy',
        'dghs-imgutils[gpu]',
        'timm',
        'aesthetic-predictor-v2-5',
        'requests',
        'faiss-cpu'
    ]

    # 安装 PyTorch 和 torchvision
    if platform.system() == 'Windows':
        if sys.version_info[:2] == (3, 11):
            torch_url = "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl"
            torchvision_url = "https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp311-cp311-win_amd64.whl"
        elif sys.version_info[:2] == (3, 10):
            torch_url = "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp310-cp310-win_amd64.whl"
            torchvision_url = "https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp310-cp310-win_amd64.whl"
    else:
        if sys.version_info[:2] == (3, 11):
            torch_url = "https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-linux_x86_64.whl"
            torchvision_url = "https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp311-cp311-linux_x86_64.whl"
        elif sys.version_info[:2] == (3, 10):
            torch_url = "https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp310-cp310-linux_x86_64.whl"
            torchvision_url = "https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp310-cp310-linux_x86_64.whl"

    if not is_package_installed("torch", activate_command):
        subprocess.run(f"{activate_command}pip install {torch_url}", shell=True, check=True)
    
    if not is_package_installed("torchvision", activate_command):
        subprocess.run(f"{activate_command}pip install {torchvision_url}", shell=True, check=True)

    # 安装 FlashAttention
    if platform.system() == 'Windows':
        if sys.version_info[:2] == (3, 11):
            flash_url = "https://github.com/oobabooga/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.2.2cxx11abiFALSE-cp311-cp311-win_amd64.whl"
        elif sys.version_info[:2] == (3, 10):
            flash_url = "https://github.com/oobabooga/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.2.2cxx11abiFALSE-cp310-cp310-win_amd64.whl"
    else:
        flash_url = "flash-attn==2.5.9.post1"

    if not is_package_installed("flash-attn", activate_command):
        subprocess.run(f"{activate_command}pip install {flash_url}", shell=True, check=True)

    for package in common_packages:
        base_package_name = package.split('==')[0].split('[')[0]
        if not is_package_installed(base_package_name, activate_command):
            subprocess.run(f"{activate_command}pip install {package}", shell=True, check=True)
    return activate_script

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
    if not os.path.exists(weight_path):
        print(f"Downloading weight file to {weight_path}...")
        response = requests.get(weight_url, stream=True)
        if response.status_code == 200:
            with open(weight_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download weight file from {weight_url}")

activate_script = install_packages()
setup_longclip()
