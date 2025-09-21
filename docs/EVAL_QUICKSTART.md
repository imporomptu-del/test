# Quickstart — Test-set Evaluation (no notebooks)

1) Activate env (VM or local):  
   `python3 -m venv ~/.venv-seaqr && source ~/.venv-seaqr/bin/activate && pip install "ultralytics>=8.3,<9" numpy opencv-python-headless matplotlib pandas pyyaml`
2) Set variables:  
   `export RUN_ID=yolo_run_20250919_212059`  
   `export WEIGHTS=gs://seaqr-ml-seaqr-detection-123-us-central1/backups/weights/best.pt`  
   `export DATASET_YAML=/data/dataset/data.yaml`    # VM path (or local path if you copied test split)
3) Run eval: `bash scripts/val_test.sh`
4) Publish (optional):  
   `export PUBLISH=gs://seaqr-detection-data-roman/test_results/$RUN_ID/ && bash scripts/publish.sh`
Outputs appear under `outputs/$RUN_ID/` → `metrics.json`, `plots/`, `results.csv`.
