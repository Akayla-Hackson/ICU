import os
import pandas as pd
from PIL import Image

def convert_annotations(data_dir, class_id=0):
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(data_dir, f'bball_{split}')
        for video_folder in sorted(os.listdir(img_dir)):
            video_path = os.path.join(img_dir, video_folder, 'img1')
            gt_path = os.path.join(img_dir, video_folder, 'gt', 'gt.txt')
            
            # Read ground truth data
            gt_data = pd.read_csv(gt_path, header=None, delimiter=',')
            
            # Filter out columns and rename
            gt_data = gt_data.iloc[:, :6]
            gt_data.columns = ['frame', 'id', 'x', 'y', 'w', 'h']

            # Group by frame to process all objects in each frame
            grouped = gt_data.groupby('frame')

            for frame_idx, boxes in grouped:
                frame_file_name = f"{str(frame_idx).zfill(6)}.jpg"
                frame_file_path = os.path.join(video_path, frame_file_name)

                if os.path.exists(frame_file_path):
                    img = Image.open(frame_file_path)
                    img_w, img_h = img.size

                    # Prepare label file content
                    label_content = []
                    for _, row in boxes.iterrows():
                        x_center = (row['x'] + row['w'] / 2) / img_w
                        y_center = (row['y'] + row['h'] / 2) / img_h
                        width = row['w'] / img_w
                        height = row['h'] / img_h

                        # # Keep player id for tracking puposes, later use
                        # player_id = row['id']
                        # label_content.append(f"{class_id} {x_center} {y_center} {width} {height} {player_id}\n")
                        label_content.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

                    # Write label file
                    label_file_name = f"{str(frame_idx).zfill(6)}.txt"
                    label_dir = os.path.join(data_dir, f'bball_{split}', video_folder, 'labels')
                    os.makedirs(label_dir, exist_ok=True)
                    label_file_path = os.path.join(label_dir, label_file_name)

                    with open(label_file_path, 'w') as label_file:
                        label_file.writelines(label_content)

convert_annotations('./data/')
