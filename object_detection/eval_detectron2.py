import torch, detectron2
import argparse
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)



def load_ground_truth(gt_path):
    ground_truths = {}
    with open(gt_path, 'r') as file:
        for line in file:
            frame_number, obj_id, x, y, w, h, _, _, _ = map(int, line.strip().split(','))
            if frame_number not in ground_truths:
                ground_truths[frame_number] = []
            ground_truths[frame_number].append((obj_id, x, y, w, h)) 
    return ground_truths

# Calculate Intersection over Union (IoU)
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
    # plt.grid(True)
    # plt.show()
    plt.savefig(save_file)


def main(args):
    data_path = "./data/bball_test"
    output_base_path = "./object_detection/outputs/updated_data_splits/detectron2"
    gt_format = 'gt.txt'

    # Ensure base output path exists
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
     # Load the trained weights
    if args.weights:
        weight_base_path = os.path.join(args.weights, 'last_checkpoint')
        # checkpoint_file = 'last_checkpoint.txt'  # Adjust the path if necessary
        if os.path.exists(weight_base_path):
            with open(weight_base_path, 'r') as file:
                checkpoint_path = file.read().strip() 
                model_path = os.path.join(args.weights, checkpoint_path)
                cfg.MODEL.WEIGHTS = model_path
        else:
            raise FileNotFoundError(f"{weight_base_path} does not exist")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    model = DefaultPredictor(cfg)
        


    all_detections = []  # Store all detections for the entire dataset
    n_global_ground_truths = 0  
    total_tp_global = 0
    total_fp_global = 0
    total_fn_global = 0

    # Process each video directory
    for video_name in sorted(os.listdir(data_path)):
        video_path = os.path.join(data_path, video_name)
        frame_directory = os.path.join(video_path, 'img1')
        gt_path = os.path.join(video_path, 'gt', gt_format)
        output_path = os.path.join(output_base_path, video_name)
        ground_truths = load_ground_truth(gt_path) if os.path.exists(gt_path) else {}

        # Ensure video-specific output path exists
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        video_detections = []
        n_ground_truths = 0
        
        with open(os.path.join(output_path, 'predictions.txt'), 'w') as predictions_file:
            # Process each frame in the video directory
            for frame_file in sorted(os.listdir(frame_directory)):
                frame_path = os.path.join(frame_directory, frame_file)
                frame_number = int(frame_file.split('.')[0])
                output_file = os.path.join(output_path, frame_file)

                # Read, process, and save the frame
                image = cv2.imread(frame_path)
                outputs = model(image)

                # Filter predictions to keep only those for the "person" class (class ID 0 in COCO)
                instances = outputs["instances"]
                person_indices = instances.pred_classes == 0
                person_instances = instances[person_indices]

                v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            
                # VISUALIZE ALL CLASSES
                # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                # # Convert Detectron2 predictions to a comparable format with ground truth
                # pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy() if outputs["instances"].has("pred_boxes") else []
                # pred_scores = outputs["instances"].scores.cpu().numpy() if outputs["instances"].has("scores") else []

                # ONLY VISUALIZE THE PERSONS CLASS
                out = v.draw_instance_predictions(person_instances.to("cpu"))
                # Convert Detectron2 predictions to a comparable format with ground truth
                pred_boxes = person_instances.pred_boxes.tensor.cpu().numpy() if person_instances.has("pred_boxes") else []
                pred_scores = person_instances.scores.cpu().numpy() if person_instances.has("scores") else []

                cv2.imwrite(output_file, out.get_image()[:, :, ::-1])

                # Get ground truth boxes for the current frame
                true_boxes = [(x, y, x + w, y + h) for _, x, y, w, h in ground_truths[frame_number]]
                n_ground_truths += len(true_boxes)


                # Calc all IoUs and store them with indices
                iou_scores = []
                for i, (pred_box, pred_score) in enumerate(zip(pred_boxes, pred_scores)):
                    pred_box_int = pred_box.astype(int)
                    for j, true_box in enumerate(true_boxes):
                        iou = calculate_iou(pred_box_int, true_box)
                        iou_scores.append((iou, i, j))  # Store IoU score along with indices
                    x, y, w, h = map(int, [pred_box_int[0], pred_box_int[1], pred_box_int[2], pred_box_int[3]])
                    predictions_file.write(f'{frame_number},{i},{x},{y},{w},{h},{pred_score:.2f},1,1\n')

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

        # Sort and calculate precision and recall for the dataset
        video_detections.sort(key=lambda x: x[0], reverse=True)
        score, matches = zip(*video_detections)
        matches = np.array(matches) 
        true_positives = np.sum(matches)  
        false_positives = len(matches) - true_positives  
        false_negatives = n_ground_truths - true_positives  

        print(f'Video {video_name} Totals. GT, matches, TP, FP, FN: {n_ground_truths}, {len(matches)}, {true_positives}, {false_positives}, {false_negatives}')

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
        print(f'Aggregate Precision, AP, Recall, Accuracy, F1 for {video_name}: {precision:.2f}, {average_precision:.2f}, {recall:.2f},  {accuracy_per_video:.2f}, {f1_score_per_video:.2f}')

        save_file = os.path.join(output_path, 'precision_recall_curve.png')
        plot_precision_recall_curve(recalls, precisions, average_precision, save_file, title=f"Precision-Recall Curve for {video_name}")

        # Add current video's detections to the global list
        all_detections.extend(video_detections)
        n_global_ground_truths += n_ground_truths
        total_tp_global += true_positives
        total_fp_global += false_positives
        total_fn_global += false_negatives


    # Sort and calculate precision and recall for the entire dataset
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

    # Calculate precision and recall at each threshold
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
    # dataset_ap = compute_ap(final_recall, final_precision)
    print(f'Dataset Precision, AP, Recall, Accuracy: {final_precision:.2f}, {dataset_ap:.2f}, {final_recall:.2f}, {global_accuracy:.2f}, {f1_score_global:.2f}')

    global_save_file = os.path.join(output_base_path, 'global_precision_recall_curve.png')
    plot_precision_recall_curve(recalls, precisions, dataset_ap, global_save_file, title="Precision-Recall Curve for Entire Dataset")


# Call the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='Path to the results file to load IE: finetuned_detectron_og')
    args = parser.parse_args()
    main(args)