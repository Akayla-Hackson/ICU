import os
import shutil
import random

def split_directories(source_directory, train_directory, val_directory, test_directory, train_count=20, val_count=5):
    # Ensure the train, validation, and test directories exist
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # List all directories in the source directory
    all_directories = [d for d in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, d))]
    random.shuffle(all_directories)

    # Assign directories to train, validation, and test sets
    train_directories = all_directories[:train_count]
    val_directories = all_directories[train_count:train_count + val_count]
    test_directories = all_directories[train_count + val_count:]

    # Copy directories to their respective new folders
    for directory in train_directories:
        shutil.copytree(os.path.join(source_directory, directory), os.path.join(train_directory, directory))

    for directory in val_directories:
        shutil.copytree(os.path.join(source_directory, directory), os.path.join(val_directory, directory))

    for directory in test_directories:
        shutil.copytree(os.path.join(source_directory, directory), os.path.join(test_directory, directory))


source_directory = './data/bball_data/'
train_directory = './data/bball_train/'
val_directory = './data/bball_val/'
test_directory = './data/bball_test/'

split_directories(source_directory, train_directory, val_directory, test_directory)

