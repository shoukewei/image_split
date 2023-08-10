import os
import random
import shutil

def train_val_test_split(input_dir, output_dir, train_ratio, test_ratio, seed, val_ratio=None):
    categories = os.listdir(input_dir)
    random.seed(seed)
    random.shuffle(categories)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    if val_ratio:
        os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)

    total_images = 0

    for category in categories:
        category_dir = os.path.join(input_dir, category)
        images = os.listdir(category_dir)
        num_images = len(images)
        total_images += num_images
        
        train_split = int(num_images * train_ratio)
        test_split = int(num_images * (train_ratio + test_ratio))
        
        train_images = images[:train_split]
        test_images = images[train_split:test_split]
        val_images = images[test_split:] if val_ratio else []
        
        os.makedirs(os.path.join(output_dir, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', category), exist_ok=True)
        
        if val_ratio:
            os.makedirs(os.path.join(output_dir, 'validation', category), exist_ok=True)
        
        for img in train_images:
            shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'train', category, img))
        for img in test_images:
            shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'test', category, img))
        for img in val_images:
            shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'validation', category, img))
    
    print("Total number of images:", total_images)
    print("Number of images in train set:", int(total_images * train_ratio))
    if val_ratio:
        print("Number of images in validation set:", int(total_images * val_ratio))
    print("Number of images in test set:", int(total_images * test_ratio))