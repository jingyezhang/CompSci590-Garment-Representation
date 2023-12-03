import os
import numpy as np
from PIL import Image

def process_batch(image_paths):
    sum_ = np.zeros(3)
    sum_sq = np.zeros(3)
    count = 0

    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert('RGB')
            img_np = np.array(img) / 255.0  # Scale to [0, 1]
            sum_ += img_np.sum(axis=(0, 1))
            sum_sq += (img_np ** 2).sum(axis=(0, 1))
            count += img_np.shape[0] * img_np.shape[1]

    return sum_, sum_sq, count

def calculate_mean_std(folder_path, batch_size=100):
    image_paths = []
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('_t_sparse.png'):
                    full_path = os.path.join(subdir_path, file)
                    print(full_path)
                    image_paths.append(full_path)

    n = len(image_paths)

    sum_ = np.zeros(3)
    sum_sq = np.zeros(3)
    count = 0

    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_sum, batch_sum_sq, batch_count = process_batch(batch_paths)
        sum_ += batch_sum
        sum_sq += batch_sum_sq
        count += batch_count

    mean = sum_ / count
    std = np.sqrt(sum_sq / count - mean ** 2)

    return mean, std

folder_path = '../../data/train'  # Replace with your dataset path
mean, std = calculate_mean_std(folder_path)
print("Mean:", mean)
print("Standard Deviation:", std)
