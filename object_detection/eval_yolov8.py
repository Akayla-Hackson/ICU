from yolo_utils import calculate_iou, detect_objects_yolo, load_ground_truth, compute_ap, plot_precision_recall_curve
from ultralytics import YOLO
import torch
import os
from PIL import Image
import argparse
import numpy as np

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
    base_output_dir = './object_detection/outputs/updated_data_splits'
    output_base_path = "./object_detection/outputs/updated_data_splits/yolov8"
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

        video_detections = []  # [(score, true_positive)]
        n_ground_truths = 0  # Total number of ground truth instances
        
        if os.path.isdir(video_path) and os.path.isfile(gt_path):
            ground_truths = load_ground_truth(gt_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

            video_detections = []  
            n_ground_truths = 0

            with open(os.path.join(output_path, 'predictions.txt'), 'w') as predictions_file:
                # Process each frame
                for frame_file in sorted(os.listdir(frames_directory)):
                    frame_number = int(frame_file.split('.')[0].lstrip("0"))  # Remove leading zeros
                    frame_path = os.path.join(frames_directory, frame_file)
                    output_file = os.path.join(output_path, frame_file)
                    if os.path.isfile(frame_path) and frame_number in ground_truths:
                        image = Image.open(frame_path).convert("RGB")

                        # Detect objects with YOLO
                        pred_boxes, predicted_image, scores = detect_objects_yolo(model, image)
                    
                        # Save the image with bounding boxes
                        if args.weights:
                            output_dir = os.path.join(base_output_dir, 'predictions_' + model_name, video_dir)
                        else:
                            output_dir = os.path.join(base_output_dir, model_name, video_dir)
                        os.makedirs(output_dir, exist_ok=True)          
                        predicted_image.save(os.path.join(output_dir, frame_file))


                        # Get ground truth boxes for the current frame
                        true_boxes = [(x, y, x + w, y + h) for _, x, y, w, h in ground_truths[frame_number]]
                        n_ground_truths += len(true_boxes)


                        # Calc all IoUs and store them with indices
                        iou_scores = []
                        for i, pred_box in enumerate(pred_boxes):
                            score = scores[i].item()
                            pred_box_np = pred_box[:4].cpu().numpy().astype(int)
                            for j, true_box in enumerate(true_boxes):
                                iou = calculate_iou(pred_box_np, true_box)
                                iou_scores.append((iou, i, j))  # Store IoU score along with indices
                            x, y, w, h = map(int, [pred_box_np[0], pred_box_np[1], pred_box_np[2], pred_box_np[3]])
                            predictions_file.write(f'{frame_number},{i},{x},{y},{w},{h},{score:.2f},1,1\n')

                        # Sort by IoU score in descending order
                        iou_scores.sort(reverse=True, key=lambda x: x[0])
                        frame_detections = [(scores[i].item(), False) for i in range(len(pred_boxes))]

                        # Select matches ensuring uniqueness
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

            for match in matches:
                tp_cumulative += match
                fp_cumulative += (1 - match)  # Increment FP if match is False
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='Path to the results file to load')
    args = parser.parse_args()
    main(args)


#example weights to load:
# ./runs/detect/finetuned_yolov8_og/weights/best.pt
