import os
import subprocess
import sys
from pathlib import Path

def create_directories():
    dirs = [
        "data/raw",
        "data/staging",
        "data/processed",
        "data/sample",
        "docs",
        "models/artifacts",
        "outputs/eda",
        "outputs/reports",
        "scripts",
        "src/haui_student_prediction",
        "src/haui_student_prediction/config",
        "src/haui_student_prediction/data",
        "src/haui_student_prediction/scoring",
        "src/haui_student_prediction/features",
        "src/haui_student_prediction/ml",
        "src/haui_student_prediction/risk",
        "src/haui_student_prediction/eda",
        "src/haui_student_prediction/services",
        "src/haui_student_prediction/storage",
        "src/haui_student_prediction/ui",
        "src/haui_student_prediction/ui/views",
        "tests/fixtures",
        "tests/unit",
        "tests/integration",
        "tests/contracts"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        
        if d.startswith("src") or d.startswith("tests"):
            init_file = Path(d) / "__init__.py"
            init_file.touch(exist_ok=True)

def setup_virtual_environment():
    venv_dir = Path(".venv")
    if not venv_dir.exists():
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)

def main():
    create_directories()
    setup_virtual_environment()

if __name__ == "__main__":
    main()