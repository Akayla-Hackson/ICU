import os
import matplotlib.pyplot as plt
import configparser

def extract_seq_length(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return int(config['Sequence']['seqLength'])

def plot_seq_lengths(data_type, lengths, directories):
    highest = max(lengths)
    lowest = min(lengths)
    average = sum(lengths) / len(lengths)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(lengths)), lengths, tick_label=directories)
    plt.xlabel('Videos')
    plt.ylabel('Sequence Length')
    plt.xticks(rotation=45, ha="right",fontsize=8)

    legend_text = f"Highest: {highest}, Lowest: {lowest}, Average: {average:.2f}"
    plt.legend([legend_text], loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{data_type}.png")

def main(data_type, source_directory):
    seq_lengths = []
    directories = []

    for subdir, dirs, files in os.walk(source_directory):
        for file in files:
            if file == 'seqinfo.ini':
                file_path = os.path.join(subdir, file)
                seq_length = extract_seq_length(file_path)
                seq_lengths.append(seq_length)
                directories.append(os.path.basename(subdir))
                print(f"Directory: {subdir}, seqLength: {seq_length}")

    plot_seq_lengths(data_type, seq_lengths, directories)

# All data
source_directory = './data/bball_data'
data_type = "all_bball"
main(data_type, source_directory)

# Train data
source_directory = './data/bball_train'
data_type = "train_bball"
main(data_type, source_directory)

# All data
source_directory = './data/bball_val'
data_type = "val_bball"
main(data_type, source_directory)

# Test data
source_directory = './data/bball_test'
data_type = "test_bball"
main(data_type, source_directory)
