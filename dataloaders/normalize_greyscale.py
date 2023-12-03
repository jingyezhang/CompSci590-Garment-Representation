import os
import numpy as np
from PIL import Image

def process_batch(image_paths):
    sum_ = 0
    sum_sq = 0
    count = 0

    for path in image_paths:
        with Image.open(path) as img:
            # Convert image to grayscale
            img = img.convert('L')
            img_np = np.array(img) / 255.0  # Scale to [0, 1]
            sum_ += img_np.sum()
            sum_sq += (img_np ** 2).sum()
            count += img_np.size  # Total number of pixels

    return sum_, sum_sq, count

def calculate_mean_std(folder_path, batch_size=100):
    image_paths = []
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('_t.png'):
                full_path = os.path.join(subdir, file)
                image_paths.append(full_path)

    n = len(image_paths)

    sum_ = 0
    sum_sq = 0
    count = 0

    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_sum, batch_sum_sq, batch_count = process_batch(batch_paths)
        sum_ += batch_sum
        sum_sq += batch_sum_sq
        count += batch_count

    # Calculate mean and std
    mean = sum_ / count
    std = np.sqrt((sum_sq / count) - (mean ** 2))

    return mean, std

folder_path = '../../data/train'  # Replace with your dataset path
mean, std = calculate_mean_std(folder_path)
print("Mean:", mean)
print("Standard Deviation:", std)
