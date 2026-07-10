"""
launch_gui.py
─────────────────────────────────────────────────────────────────────────────
CRC-AI Clinical GUI — launcher

Run from VS Code terminal (project root):
    python launch_gui.py

Requirements beyond requirements.txt:
    pip install Pillow
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    try:
        import tkinter
    except ImportError:
        print("ERROR: tkinter is not available.")
        print("On Ubuntu/Debian: sudo apt-get install python3-tk")
        print("On Windows/macOS: tkinter is bundled with Python.")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("Installing Pillow...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "Pillow", "--quiet"])

    from gui.app import launch
    launch()
