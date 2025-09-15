yolo detect train \
    model=/mnt/d/KOLOMVERSE/KOLOMVERSE/scripts/models/yolov8s_kolomverse/weights/best.pt \
    data=models/kolo_whales.yaml \
    imgsz=640 \
    epochs=13 \
    batch=8 \
    lr0=0.001 \
    freeze=10 \
    project=runs/finetune \
    name=whales