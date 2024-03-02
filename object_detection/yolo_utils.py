import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Object detection with YOLO
def detect_objects_yolo(model, image):
    results = model(image, verbose=False)
    # results = model(image, verbose=False, classes=0)
    # Extract the boxes, confidences, and class labels
    for result in results:
        boxes = result.boxes.xyxy
        im_array = result.plot()
        predicted_image = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        scores = result.boxes.conf
    return boxes, predicted_image, scores

# Function to parse the ground truth data
def load_ground_truth(gt_path):
    ground_truths = {}
    with open(gt_path, 'r') as file:
        for line in file:
            frame_number, obj_id, x, y, w, h, _, _, _ = map(int, line.strip().split(','))
            if frame_number not in ground_truths:
                ground_truths[frame_number] = []
            ground_truths[frame_number].append((obj_id, x, y, w, h))
    return ground_truths

# Ccalculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    # print("pred box:", boxA)
    # print("true box:", boxB)
    # Convert boxes to the format (x1, y1, x2, y2)
    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
    
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves."""
    # First append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # And sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def plot_precision_recall_curve(recall, precision, ap, save_file, title="Precision-Recall Curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, marker='.', label=f'AP = {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.savefig(save_file)