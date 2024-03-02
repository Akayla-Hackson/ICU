import os
import shutil

def read_directory_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def copy_directories(source_directory, destination_directory, directory_names):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for dir_name in directory_names:
        source_path = os.path.join(source_directory, dir_name)
        dest_path = os.path.join(destination_directory, dir_name)

        if os.path.exists(source_path) and os.path.isdir(source_path):
            shutil.copytree(source_path, dest_path)
            print(f"Copied {dir_name} to {destination_directory}")
        else:
            print(f"Directory not found: {dir_name}")


txt_file_path = './data/sportsmot_publish/splits_txt/basketball.txt' 
source_directory = './data/sportsmot_publish/dataset/combined/' 
destination_directory = './data/bball_data/'

directory_names = read_directory_names(txt_file_path)
copy_directories(source_directory, destination_directory, directory_names)
