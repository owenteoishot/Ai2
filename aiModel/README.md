# AI Model Webapp
A small web frontend + backend to run the pose-classification model from webcam frames.

## Quick Start

# create & activate venv (from root folder)
python -m venv .venv
.\.venv\Scripts\activate

# install deps(from root .venv)
pip install -r aiModel/requirements.txt

# ensure uvicorn websocket support (inside venv)
pip install "uvicorn[standard]"

# start with uvicorn
uvicorn aiModel.webapp.backend:app --host 127.0.0.1 --port 8000 --reload

# open frontend
 http://127.0.0.1:8000/static/index.html


you will have to wait for a minute or so for the model to warm up

## Slow Start

Windows (cmd.exe)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r aiModel/requirements.txt
```

Start server (script)

```bash
python aiModel/webapp/backend.py --host 127.0.0.1 --port 8000 --model aiModel/models/ai_charades_tcn2.pt
```

Start server (uvicorn)

```bash
uvicorn aiModel.webapp.backend:app --host 127.0.0.1 --port 8000 --reload
```

Note: ensure websockets support:

```bash
pip install "uvicorn[standard]"
```

Open the frontend:

```text
http://127.0.0.1:8000/static/index.html
```

macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r aiModel/requirements.txt
```

Set env vars and run with uvicorn

Windows (cmd):

```bash
set AI_MODEL_PATH=aiModel/models/ai_charades_tcn2.pt && set AI_SEQ_LEN=48 && uvicorn aiModel.webapp.backend:app --host 127.0.0.1 --port 8000 --reload
```

macOS/Linux:

```bash
AI_MODEL_PATH=aiModel/models/ai_charades_tcn2.pt AI_SEQ_LEN=48 uvicorn aiModel.webapp.backend:app --host 127.0.0.1 --port 8000 --reload
```

## Troubleshooting

- WebSocket 404 or "Unsupported upgrade": install uvicorn extras and restart:

```bash
pip install "uvicorn[standard]"
```

- Frontend shows "model_not_loaded": ensure the model is loaded at server startup or set AI_MODEL_PATH and restart. Example restart:

Windows:

```bash
set AI_MODEL_PATH=aiModel/models/ai_charades_tcn2.pt && uvicorn aiModel.webapp.backend:app --host 127.0.0.1 --port 8000 --reload
```

macOS/Linux:

```bash
AI_MODEL_PATH=aiModel/models/ai_charades_tcn2.pt uvicorn aiModel.webapp.backend:app --host 127.0.0.1 --port 8000 --reload
```

- Check server logs in the terminal where you started the script/uvicorn. Test health:

```bash
curl http://127.0.0.1:8000/health
```