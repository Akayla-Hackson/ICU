import os
import cv2
import torch
from PIL import ImageDraw
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, FasterRCNN_ResNet50_FPN_Weights , SSD300_VGG16_Weights
from PIL import Image
from ultralytics import YOLO
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Object detection
def detect_objects(model, image):
    transform = T.Compose([T.ToTensor()])
    image = transform(image).unsqueeze(0)  # add a batch dimension
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        prediction = model(image)
    # Move each tensor in the prediction to CPU
    for pred in prediction:
        pred['boxes'] = pred['boxes'].cpu()
        pred['labels'] = pred['labels'].cpu()
        pred['scores'] = pred['scores'].cpu()
    return prediction


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

# Draw bounding boxes on the image
def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        draw.text((box[0], box[1]), str(label), fill="red")
    return image


# Ccalculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
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

# Compute the average precision
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
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


def main(args):

    # Load the model
    if args.model == 'faster_RCNN':
        if args.weights:
            model = fasterrcnn_resnet50_fpn(weights=args.weights).eval()
            model_name = 'finetuned_fasterrcnn'
        else:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
            model_name = 'fasterrcnn'
    elif args.model == 'SSD':
        if args.weights:
            model = ssd300_vgg16(weights=args.weights).eval()
            model_name = 'finetuned_ssd'
        else:
            model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).eval()
            model_name = 'ssd'
    else:
        print("Model is not available")
        exit()
                        
    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA")

    # Test videos directory
    test_videos_directory = './data/bball_test/'
    base_output_dir = './object_detection/outputs/updated_data_splits'
    output_base_path = os.path.join(base_output_dir, model_name)

    # Ensure base output path exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path, exist_ok=True)
    
    all_detections = [] 
    n_global_ground_truths = 0  
    total_tp_global = 0
    total_fp_global = 0
    total_fn_global = 0


    # Process each video directory
    for video_dir in sorted(os.listdir(test_videos_directory)):
        video_path = os.path.join(test_videos_directory, video_dir)
        gt_path = os.path.join(video_path, 'gt', 'gt.txt')
        frames_directory = os.path.join(video_path, 'img1')
        output_path = os.path.join(output_base_path, video_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        video_detections = []  
        n_ground_truths = 0  

        if os.path.isdir(video_path) and os.path.isfile(gt_path):
            ground_truths = load_ground_truth(gt_path)

            # Process each frame
            for frame_file in sorted(os.listdir(frames_directory)):
                frame_number = int(frame_file.split('.')[0].lstrip("0"))  # Remove leading zeros
                frame_path = os.path.join(frames_directory, frame_file)
                if os.path.isfile(frame_path) and frame_number in ground_truths:
                    # Load the image
                    image = Image.open(frame_path).convert("RGB")

                    # Detect objects
                    prediction = detect_objects(model, image)

                    # Unpack predictions
                    pred_boxes = prediction[0]["boxes"].detach().numpy()
                    pred_scores = prediction[0]["scores"].detach().numpy()
                    pred_labels = prediction[0]["labels"].detach().numpy()

                    # Filter out predictions based on score threshold (e.g., 0.5)
                    threshold = 0.5
                    pred_boxes = pred_boxes[pred_scores >= threshold]
                    pred_labels = pred_labels[pred_scores >= threshold]

                    # Convert boxes to (x1, y1, x2, y2) format for drawing
                    pred_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in pred_boxes]

                    # Draw bounding boxes on the image
                    drawn_image = draw_boxes(image, pred_boxes, pred_labels)

                    # Save the image with bounding boxes
                    drawn_image.save(os.path.join(output_path, frame_file))

                    true_boxes = [(x, y, x + w, y + h) for _, x, y, w, h in ground_truths[frame_number]]

                    n_ground_truths += len(true_boxes)

                    iou_scores = []
                    for i, (pred_box, pred_score) in enumerate(zip(pred_boxes, pred_scores)):
                        pred_box_int = np.array(pred_box).astype(int)
                        for j, true_box in enumerate(true_boxes):
                            iou = calculate_iou(pred_box_int, true_box)
                            iou_scores.append((iou, i, j))  # Store IoU score along with indice

                    # Sort by IoU score in descending order
                    iou_scores.sort(reverse=True, key=lambda x: x[0])
                    frame_detections = [(pred_scores[i].item(), False) for i in range(len(pred_boxes))]

                    # Select matches ensuring unique association
                    matched_predictions = set()
                    matched_ground_truths = set()
                    for score, pred_index, true_index in iou_scores:
                        if score <= 0.5:
                            break  # No more matches above the threshold
                        if pred_index not in matched_predictions and true_index not in matched_ground_truths:
                            matched_predictions.add(pred_index)
                            matched_ground_truths.add(true_index)
                            # Update video_detections with True for matched predictions
                            frame_detections[pred_index] = (frame_detections[pred_index][0], True)

                    video_detections.extend(frame_detections)
            

           # Sort and calculate metrics
            video_detections.sort(key=lambda x: x[0], reverse=True)
            score, matches = zip(*video_detections)
            matches = np.array(matches) 
            true_positives = np.sum(matches)  
            false_positives = len(matches) - true_positives  
            false_negatives = n_ground_truths - true_positives  

            print(f'Video {video_dir} Totals. GT, matches, TP, FP, FN: {n_ground_truths}, {len(matches)}, {true_positives}, {false_positives}, {false_negatives}')

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / n_ground_truths
            accuracy_per_video = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
            if (precision + recall) > 0:
                f1_score_per_video = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score_per_video = 0

            precisions = []
            recalls = []
            tp_cumulative = 0
            fp_cumulative = 0

            # Calculate precision and recall at each threshold
            for match in matches:
                tp_cumulative += match
                fp_cumulative += (1 - match) 
                pr = tp_cumulative / (tp_cumulative + fp_cumulative)
                re = tp_cumulative / n_ground_truths
                precisions.append(pr)
                recalls.append(re)

            # Convert lists to arrays for AP calculation
            precisions = np.array(precisions)
            recalls = np.array(recalls)


            # Calculate Average Precision (AP)
            average_precision = compute_ap(recalls, precisions)
            print(f'Aggregate Precision, AP, Recall, Accuracy, F1 for {video_dir}: {precision:.2f}, {average_precision:.2f}, {recall:.2f},  {accuracy_per_video:.2f}, {f1_score_per_video:.2f}')

            save_file = os.path.join(output_path, 'precision_recall_curve.png')
            plot_precision_recall_curve(recalls, precisions, average_precision, save_file, title=f"Precision-Recall Curve for {video_dir}")

            # Add current video's detections to the global list
            all_detections.extend(video_detections)
            n_global_ground_truths += n_ground_truths
            total_tp_global += true_positives
            total_fp_global += false_positives
            total_fn_global += false_negatives


    # Sort and calculate metrics for the entire dataset
    all_detections.sort(key=lambda x: x[0], reverse=True)
    score, matches = zip(*all_detections)
    matches = np.array(matches) 
    true_positives = np.sum(matches)  
    false_positives = len(matches) - true_positives  
    false_negatives = n_ground_truths - true_positives 

    print(f'Global Totals. GT, matches, TP, FP, FN: {n_global_ground_truths}, {len(matches)}, {total_tp_global}, {total_fp_global}, {total_fn_global}')
    final_precision = true_positives / (true_positives + false_positives)
    final_recall = true_positives / n_global_ground_truths
    global_accuracy = total_tp_global / (total_tp_global + total_fp_global + total_fn_global) if (total_tp_global + total_fp_global + total_fn_global) > 0 else 0

    if (final_precision + final_recall) > 0:
        f1_score_global = 2 * (final_precision * final_recall) / (final_precision + final_recall)
    else:
        f1_score_global = 0

    precisions = []
    recalls = []
    tp_cumulative = 0
    fp_cumulative = 0

    for match in matches:
        tp_cumulative += match
        fp_cumulative += (1 - match)  # Increment FP if match is False
        pr = tp_cumulative / (tp_cumulative + fp_cumulative)
        re = tp_cumulative / n_global_ground_truths
        precisions.append(pr)
        recalls.append(re)

    # Convert lists to arrays for AP calculation
    precisions = np.array(precisions)
    recalls = np.array(recalls)


    # Calculate Average Precision (AP)
    dataset_ap = compute_ap(recalls, precisions)

    # Calculate AP for the entire dataset
    print(f'Dataset Precision, AP, Recall, Accuracy, F1: {final_precision:.2f}, {dataset_ap:.2f}, {final_recall:.2f}, {global_accuracy:.2f}, {f1_score_global:.2f}')

    global_save_file = os.path.join(output_base_path, 'global_precision_recall_curve.png')
    plot_precision_recall_curve(recalls, precisions, dataset_ap, global_save_file, title="Precision-Recall Curve for Entire Dataset")


    

# Call the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='faster_RCNN', help='Model to load')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    main(args)
