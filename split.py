import os
import random
import shutil

def train_val_test_split(input_dir, output_dir, train_ratio, test_ratio, seed, val_ratio=None):

    """
    Split image dateset into train, validation and test sets, and automatically
    create three subfolders in the output_dir in the Currently Working Directory:
    - train
    - val
    - test

    Constructor method.
        Parameters
        ----------
        input dir: the orginal image data path
        output dir: the path to store the split data
        train_ratio: the ratio of train data number to the total data number
        val_ratio: the ratio of validation data number to the total data number, default is 0
        test_ratio: the ratio of test data number to the total data number
        
        Return
        ---------- 
        train: the train directory containing train data set
        val: the val directory containing validation data set
        test: the test directory containing test data set

        If val_ratio is not specified, i.e. default 0, it only generates train and test sets
    """
    
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