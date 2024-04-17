Команда для запуска обучения нейросети:
python ./yolov5/train.py --data *Путь к data.yaml* --cfg ./yolov5/models/yolov5s.yaml --batch-size 64 --name Model --epochs 300

!python .yolov5export.py --weight .yolov5runstrainModelweightsbest.pt --include torchscript onnx
