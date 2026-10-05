"""
Project 18: Launcher for Demo Web Application
Usage:
    python run_demo.py
    hoặc
    python Demo/app.py
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_APP = PROJECT_ROOT / "Demo" / "app.py"

if __name__ == "__main__":
    print(f"🚀 Đang khởi động Web Demo từ: {DEMO_APP}")
    subprocess.run([sys.executable, str(DEMO_APP)])
