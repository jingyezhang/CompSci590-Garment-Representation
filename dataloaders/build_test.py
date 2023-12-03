import os
import random
import shutil

def move_random_subfolders(src, dst, num_folders=500):
    """
    Moves a random selection of subfolders from src to dst.

    :param src: Source directory containing the subfolders
    :param dst: Destination directory to move subfolders to
    :param num_folders: Number of random subfolders to move
    """

    # Ensure the source directory exists
    if not os.path.exists(src):
        print(f"Source directory {src} does not exist.")
        return

    # Ensure the destination directory exists, create if it doesn't
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Get all subfolders in the source directory
    all_subfolders = [os.path.join(src, f) for f in os.listdir(src) if os.path.isdir(os.path.join(src, f))]

    # Check if there are enough subfolders
    if len(all_subfolders) < num_folders:
        print(f"Only {len(all_subfolders)} subfolders available, need {num_folders}.")
        return

    # Randomly select subfolders
    selected_subfolders = random.sample(all_subfolders, num_folders)

    # Move each selected subfolder to the destination
    for folder in selected_subfolders:
        dst_folder = os.path.join(dst, os.path.basename(folder))
        shutil.move(folder, dst_folder)

    print(f"Moved {num_folders} subfolders to {dst}.")

# Example usage
src_directory = "../../data/dataset"  # Replace with your source directory path
dst_directory = "../../data/test"  # Replace with your destination directory path

move_random_subfolders(src_directory, dst_directory)