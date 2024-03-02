from yolo_utils import calculate_iou, detect_objects_yolo, load_ground_truth
from ultralytics import YOLO
import torch
import os
from PIL import Image
import argparse


def main(args):

    # Load the trained weights
    if args.weights:
        model = YOLO(args.weights) 
        model_name = 'yolov8'
    else:
        # Load the model
        model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
        model_name = 'yolov8'

    # Check for CUDA
    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA")

    # Test videos directory
    test_videos_directory = './data/bball_test/'
    base_output_dir = './object_detection/outputs'

    all_precision_scores = []
    all_recall_scores = []

    # Evaluation
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

                    # Detect objects with YOLO
                    pred_boxes, predicted_image = detect_objects_yolo(model, image)
                  
                    # Save the image with bounding boxes
                    if args.weights:
                        output_dir = os.path.join(base_output_dir, 'weights_' + model_name, video_dir)
                    else:
                        output_dir = os.path.join(base_output_dir, model_name, video_dir)
                    os.makedirs(output_dir, exist_ok=True)          
                    predicted_image.save(os.path.join(output_dir, frame_file))


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
    parser.add_argument('--weights', type=str, default=None, help='Path to the results file to load')
    args = parser.parse_args()
    main(args)
