import cv2
import os

# Function to parse the ground truth file
def parse_gt_file(gt_file_path):
    boxes = {}
    with open(gt_file_path, 'r') as f:
        for line in f:
            frame, obj_id, bb_left, bb_top, bb_width, bb_height, conf, x, y = map(int, line.strip().split(','))
            if frame not in boxes:
                boxes[frame] = []
            boxes[frame].append((bb_left, bb_top, bb_width, bb_height, obj_id))
    return boxes

# Function to draw bounding boxes on an image
def draw_boxes(image, boxes):
    for bb_left, bb_top, bb_width, bb_height, obj_id in boxes:
        x_min, y_min = bb_left, bb_top
        x_max, y_max = bb_left + bb_width, bb_top + bb_height
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f'ID {obj_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def process_video(video_folder, base_output_folder):
    image_folder = os.path.join(video_folder, 'img1/')
    gt_file_path = os.path.join(video_folder, 'gt/gt.txt')

    # Check if the ground truth file and image folder exist
    if not os.path.exists(image_folder) or not os.path.isfile(gt_file_path):
        print(f"Image folder or gt.txt file does not exist for video {video_folder}. Skipping...")
        return

    ground_truths = parse_gt_file(gt_file_path)

    # Extract video name and create output directory
    video_name = os.path.basename(video_folder)
    output_folder = os.path.join(base_output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Process all frames for which ground truth data is available
    for frame_number in sorted(ground_truths.keys()):
        frame_filename = f"{str(frame_number).zfill(6)}.jpg"
        frame_path = os.path.join(image_folder, frame_filename)
        frame = cv2.imread(frame_path)

        if frame is not None:
            frame_with_boxes = draw_boxes(frame, ground_truths[frame_number])
            output_path = os.path.join(output_folder, f"frame_{frame_filename}")
            cv2.imwrite(output_path, frame_with_boxes)
        else:
            print(f"Frame {frame_path} could not be loaded. Please check the file path and format.")

    print(f"Processed and saved frames for video {video_name} in {output_folder}.")

# Usage
video_folder = './data/bball_test/v_00HRwkvvjtQ_c001'
base_output_folder = './gt_plots/'
process_video(video_folder, base_output_folder)