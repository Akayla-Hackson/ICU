import cv2
import os

def create_video_from_frames(video_folder, base_output_folder, output_video_name, frame_rate=30.0):
    video_name = os.path.basename(video_folder)
    output_folder = os.path.join(base_output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Get a sorted list of frame filenames
    frame_filenames = sorted([f for f in os.listdir(video_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Assume all frames are of the same size, get size from the first frame
    first_frame_path = os.path.join(video_folder, frame_filenames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print("Error loading the first frame.")
        return
    frame_height, frame_width, _ = first_frame.shape
    frame_size = (frame_width, frame_height)
    
    # Define the codec and create VideoWriter object
    output_video_path = os.path.join(output_folder, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    for frame_filename in frame_filenames:
        frame_path = os.path.join(video_folder, frame_filename)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Frame {frame_path} could not be loaded. Skipping frame.")

    # Release everything when job is finished
    out.release()
    print(f"Video has been processed and saved as {output_video_path}.")

# EUsage
video_folder = './object_detection/outputs/updated_data_splits/yolov8_only_ppl_class/v_00HRwkvvjtQ_c001'  
base_output_folder = './demos/'
output_video_name = 'yolov8_only_ppl_v_00HRwkvvjtQ_c001.mp4'
create_video_from_frames(video_folder, base_output_folder, output_video_name)


