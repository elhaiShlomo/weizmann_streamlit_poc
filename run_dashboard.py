import os
import sys
import subprocess
from pathlib import Path

def get_exe_real_path():
    # If running as EXE → get folder of EXE
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    # Running as .py
    return Path(__file__).parent

def main():
    # Force working directory to be the project root
    project_dir = get_exe_real_path()
    os.chdir(project_dir)

    print(f"Working directory: {project_dir}")

    # Paths inside project
    venv_python = project_dir / ".venv" / "Scripts" / "python.exe"
    streamlit_exec = project_dir / ".venv" / "Scripts" / "streamlit.exe"
    app_file = project_dir / "app.py"

    # Validation
    if not venv_python.exists():
        print("python.exe not found inside .venv")
        input("Press ENTER...")
        return

    if not streamlit_exec.exists():
        print("streamlit.exe not found inside .venv")
        input("Press ENTER...")
        return

    if not app_file.exists():
        print("app.py not found in project directory")
        input("Press ENTER...")
        return

    print("Launching dashboard...")
    print(f"Executable: {streamlit_exec}")
    print(f"App file:  {app_file}")

    subprocess.Popen([str(streamlit_exec), "run", str(app_file)])

if __name__ == "__main__":
    main()
