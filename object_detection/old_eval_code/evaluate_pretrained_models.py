import os
import cv2
import torch
from PIL import ImageDraw
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, FasterRCNN_ResNet50_FPN_Weights , SSD300_VGG16_Weights
from PIL import Image
from ultralytics import YOLO
import argparse

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




def main(args):

    # Load the model
    if args.model == 'faster_RCNN':
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
        model_name = 'fasterrcnn_resnet50_fpn'
    elif args.model == 'SSD':
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
    base_output_dir = './object_detection/outputs'

    all_precision_scores = []
    all_recall_scores = []
    
    # Process each video directory
    for video_dir in sorted(os.listdir(test_videos_directory)):
        video_path = os.path.join(test_videos_directory, video_dir)
        gt_path = os.path.join(video_path, 'gt', 'gt.txt')
        frames_directory = os.path.join(video_path, 'img1')

        
        if os.path.isdir(video_path) and os.path.isfile(gt_path):
            ground_truths = load_ground_truth(gt_path)

            total_true_positives = 0
            total_false_positives = 0
            total_false_negatives = 0

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
                    output_dir = os.path.join(base_output_dir, model_name, video_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    drawn_image.save(os.path.join(output_dir, frame_file))

                    # Get ground truth boxes for the current frame
                    true_boxes = [(x, y, x + w, y + h) for _, x, y, w, h in ground_truths[frame_number]]

                    # Calculate IoUs and match predictions to ground truth
                    matched_ground_truths = set()
                    for pred_box in pred_boxes:
                        iou_for_this_pred = [calculate_iou(pred_box, true_box) for true_box in true_boxes]
                        max_iou = max(iou_for_this_pred, default=0)
                        if max_iou > 0.5:
                            total_true_positives += 1
                            matched_ground_truths.add(iou_for_this_pred.index(max_iou))
                        else:
                            total_false_positives += 1

                    # False negatives are ground truth boxes that weren't matched
                    total_false_negatives += len(true_boxes) - len(matched_ground_truths)

                    # print(f'Frame {frame_number}: Precision = {precision:.2f}, Recall = {recall:.2f}')

            # Calculate aggregate metrics
            precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) else 0
            recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) else 0

            # Display aggregate metrics
            print(f'Aggregate Precision & Recall for {video_dir}: {precision:.2f}, {recall:.2f}')

            # Add metrics to calc average
            all_precision_scores.append(precision)
            all_recall_scores.append(recall)


    average_precision = sum(all_precision_scores)/ len(all_precision_scores)
    average_recall = sum(all_recall_scores)/ len(all_recall_scores)

    print(f'Average Precision and Recall: {average_precision:.2f}, {average_recall:.2f}')
        
    

# Call the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='faster_RCNN', help='Model to load')
    args = parser.parse_args()
    main(args)
