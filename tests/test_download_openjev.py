"""直接下载 OpenJev 到本地目录,避免 Windows symlink 问题。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

# 目标目录:项目内 models/openjev
TARGET_DIR = Path(ROOT) / "models" / "openjev"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("直接下载 OpenJev 到本地目录(无 symlink)")
print("=" * 60)
print(f"[镜像] {os.environ['HF_ENDPOINT']}")
print(f"[目标] {TARGET_DIR}")
print("-" * 60)

from huggingface_hub import snapshot_download

print("[INFO] 开始下载(直接落盘)...")
print("[INFO] 模型文件约 1.7GB,请耐心等待...")

result = snapshot_download(
    repo_id="AlexWortega/openjev",
    revision="main",
    allow_patterns=["qwen3.5-0.8b-nli-v2s-long/*"],
    local_dir=str(TARGET_DIR),
    local_dir_use_symlinks=False,  # 关键:不用 symlink,直接复制
)

print("-" * 60)
print(f"[OK] 下载完成: {result}")

# 列出下载的文件
files = list(Path(TARGET_DIR).rglob("*"))
real_files = [f for f in files if f.is_file()]
print(f"[INFO] 共 {len(real_files)} 个文件:")
total_size = 0
for f in real_files:
    rel = f.relative_to(TARGET_DIR)
    size_mb = f.stat().st_size / 1024 / 1024
    total_size += size_mb
    print(f"  - {rel} ({size_mb:.1f} MB)")
print(f"[INFO] 总大小: {total_size:.1f} MB")

print("=" * 60)
print(f"[DONE] 模型已下载到: {TARGET_DIR / 'qwen3.5-0.8b-nli-v2s-long'}")
print(f"[NEXT] 修改 jev.py 使用本地路径加载")
