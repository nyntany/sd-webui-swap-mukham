import os
import time
import datetime
import subprocess
import requests

models_dir = os.path.dirname(os.path.abspath(__file__))

def download_models_bt():
    models_download_dir = os.path.join(models_dir, "assets", "pretrained_models")
    num_urls = len(urls)
    num_downloaded = 0
    for url in urls:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            filename = os.path.basename(url)
            file_path = os.path.join(models_download_dir, filename)

            if os.path.exists(file_path):
                yield f"💬 Checking availible models."
                num_downloaded += 1
                continue

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB
            downloaded_size = 0
            prev_progress = -1

            with open(file_path, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded_size += len(data)
                    percentage = round(downloaded_size / total_size * 100, 1)
                    current_progress = int(percentage // 5)
                    
                    if current_progress > prev_progress:
                        progress_bar = "🟩" * current_progress + "" * (50 - current_progress)
                        total_size_mb = total_size / (1024 * 1024)  # Convert total_size to megabytes
                        yield f"Downloading: {filename}({total_size_mb:.2f}) MB {progress_bar} Completed: {percentage:.1f}%"
                        prev_progress = current_progress

            num_downloaded += 1

        except requests.exceptions.RequestException as e:
            yield f"💢 Couldn't download {filename}. Error: {e}"
        except Exception as e:
            yield f"💢 Unknown error downloading {filename}. Error: {e}"

    if num_downloaded == num_urls:
        yield "🆗 All Models downloaded."
    time.sleep(5)
    yield ""

urls = [
    "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "https://huggingface.co/bluefoxcreation/Codeformer-ONNX/resolve/main/codeformer.onnx",
    "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-256.onnx",
    "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-512.onnx",
    "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/restoreformer.onnx",
    "https://github.com/zllrunning/face-makeup.PyTorch/raw/master/cp/79999_iter.pth",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth",
]




