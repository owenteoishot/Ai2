AI Charades — webcam demo

Quick start (example commands)

1) Create & activate a virtual environment
# POSIX / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (cmd.exe)
python -m venv .venv
.venv\Scripts\activate

2) Install dependencies
pip install -r aiModel/requirements.txt

3) Run the webcam demo
python aiModel/run_webcam.py --model aiModel/models/ai_charades_tcn2.pt --camera 0

4) Run tests
python -m pytest

Recommended one-line launch.json snippet (add to configurations array)
{"name":"Run AI Charades","type":"python","request":"launch","program":"${workspaceFolder}/aiModel/run_webcam.py","args":["--model","${workspaceFolder}/aiModel/models/ai_charades_tcn2.pt","--camera","0"]}

Notes
- Model files (aiModel/models/*.pt) and dataset folders are large; do not commit them if you add or replace models.
- If the checkpoint fails to load, the script prints an error and exits with status 1.