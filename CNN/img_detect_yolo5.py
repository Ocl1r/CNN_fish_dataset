import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


INPUT_WIDTH = 640
INPUT_HEIGHT = 640

net = cv2.dnn.readNetFromONNX('C:/Users/Ocl1r/Desktop/flash/files/cnn_proj/nn/yolov5/runs/train/Model3/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections


def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index


def drawings(image, boxes_np, confidences_np, index):
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'fish: {:.0f}%'.format(bb_conf * 100)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (0, 255, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return image


def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img = drawings(img, boxes_np, confidences_np, index)
    return result_img


img = mpimg.imread('C:/Users/Ocl1r/Desktop/flash/files/cnn_proj/nn/data/TEST/TEST_7.jpeg')

plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Исходное изображение')

results = yolo_predictions(img, net)
plt.subplot(1, 2, 2)
plt.imshow(results)
plt.title('Изображение с меткой')

plt.show()
