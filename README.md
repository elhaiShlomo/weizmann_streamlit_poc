
# Streamlit Dashboard - Setup & Run Guide

This guide explains how to install and run the Streamlit dashboard locally.  
Initial setup is required **once**, after which you can launch the dashboard simply by double‑clicking the provided EXE.

---

## 1. Requirements

Make sure you have:

- **Python 3.10+** installed  
  Download: https://www.python.org/downloads/  
  *During installation, check: “Add Python to PATH”*

---

## 2. One-Time Setup

### 1. Open PowerShell inside the project folder

Right-click the folder → **Open PowerShell window here**  
or:

```powershell
cd "C:\path\to\weizmann_streamlit_poc"
```

### 2. Allow script execution (required by Windows)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### 3. Create a virtual environment

```powershell
python -m venv .venv
```

### 4. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install all required dependencies

```powershell
pip install -r requirements.txt
```

This step may take a few minutes.  
After it finishes — setup is complete.

---

## 3. Running the Dashboard (Daily Use)

Once setup is done, simply double‑click:

```
run_dashboard.exe
```

This automatically starts Streamlit and opens the dashboard at:

```
http://localhost:8501
```

You do **not** need to run any commands after the first installation.

---

## 4. Troubleshooting

### ⚠ “.venv not found” or missing dependencies
Activate environment and reinstall:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### ⚠ Windows blocks script execution
Run:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope Process
```

### ⚠ Python not recognized
Reinstall Python and ensure “Add Python to PATH” is checked.

---

## 5. You're Ready to Go!
If you encounter issues, contact the project owner.
