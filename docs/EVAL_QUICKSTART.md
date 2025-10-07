# Eval Quickstart
1) `python3.11 -m venv .venv311 && source .venv311/bin/activate && pip install -r requirements.txt`
2) Copy your local dataset YAML to `configs/data.local.yaml` (see example file).
3) Put `best.pt` at `../runs/train/<run>/weights/best.pt` or set MODEL env var.
4) Run: `yolo val model=../runs/train/.../best.pt data=configs/data.local.yaml split=val batch=2 imgsz=640 device=cpu save_json=True plots=True`
Outputs land under `outputs/<yolo_run_*>/`.
