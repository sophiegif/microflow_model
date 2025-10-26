import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import os
from src.dataloaders.tiff_utils import open_tiff, get_tiff_band, get_tiff_crs, get_tiff_transform
import torch
import rasterio as rio
from rasterio.windows import Window
import math
from src.dataloaders.tiff_utils import save_array_to_tiff
import time
import matplotlib.pyplot as plt

class RealExampleLargeDataset(Dataset):

    def __init__(
            self, root_dir, transform=None, pre_template="pre.tif", post_template="post.tif", top=0, left=0,
            window_size=1024, window_overlap=64
    ):
        self.root_dir = root_dir  # contains all the input-pair folders
        self.transform = transform
        self.pre_template = pre_template
        self.post_template = post_template
        self.top = top
        self.left = left
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.height = -1
        self.width = -1
        self.image_pair_name = None
        self.pre_filename, self.post_filename = self.get_filenames(self.root_dir)

        # load image info from rasterio (efficiently implemented)
        with rio.open(self.pre_filename) as pre_raster:
            self.width = pre_raster.width
            self.height = pre_raster.height
            self.crs_info = pre_raster.crs
            self.transform_info = pre_raster.transform
        print(f"self.width {self.width} self.height {self.height}")

        t0 = time.time()
        self.load_patches_in_memory()
        t1 = time.time()
        print(f"time {t1 - t0}s")

    def load_and_fill_array(self, filename):
        """ Fill nan values with 0 because it is computationnally expensive to do a nearest ND interpolation for large images.
        If needed otherwise, best practice would be to feed input images without nan values (or preprocess it)
        """
        image = open_tiff(filename)
        array = get_tiff_band(image, 1)
        return array

    def load_prepost_array(self, pre_filename, post_filename):
        """ Loads prepost tensor """
        pre_array = self.load_and_fill_array(pre_filename)
        post_array = self.load_and_fill_array(post_filename)
        return np.array([pre_array, post_array])

    def load_patches_in_memory(self):
        pre_filename, post_filename = self.get_filenames(self.root_dir)
        self.pre_path = pre_filename
        self.post_path = post_filename

        pre_post = self.load_prepost_array(pre_filename, post_filename)

        # Extract crs metadata
        pre_img = open_tiff(pre_filename)
        self.crs_info = get_tiff_crs(pre_img)
        self.transform_info = get_tiff_transform(pre_img)

        # Set image size
        self.height = pre_post.shape[-2]
        self.width = pre_post.shape[-1]

        # Extract the patches in memory
        self.pre_posts, self.x_positions, self.y_positions = self.extract_patches(pre_post)

    def extract_patches(self, images):
        def update_patches(images, x, y, window_size, patches, x_positions, y_positions):
            patch = images[:, y: y + window_size, x: x + window_size]
            if np.sum(np.abs(patch)) == 0:  # for regions where there is no data
                return
            patches.append(patch)
            x_positions.append(x)
            y_positions.append(y)

        _, h, w = images.shape
        patches, x_positions, y_positions = [], [], []

        stride = self.window_overlap
        for y in range(0, h - self.window_size + stride, stride):
            for x in range(0, w - self.window_size + stride, stride):
                # Handling edge cases
                if y + self.window_size > h:
                    y = h - self.window_size
                if x + self.window_size > w:
                    x = w - self.window_size
                update_patches(images, x, y, self.window_size, patches, x_positions, y_positions)

        return np.array(patches), x_positions, y_positions

    def get_filenames(self, example_dir):
        filenames = os.listdir(example_dir)
        pre_filename, post_filename = None, None
        for filename in filenames:
            if filename.endswith(self.pre_template):
                pre_filename = os.path.join(example_dir, filename)
                self.image_pair_name = filename.replace(self.pre_template, "")
            if filename.endswith(self.post_template):
                post_filename = os.path.join(example_dir, filename)

        return pre_filename, post_filename


    def __len__(self):
        return len(self.pre_posts)

    def get_image_info(self):
        """ Get the tiff metadata, which are not at the right format to be yield by the dataloader """
        return self.height, self.width, self.window_overlap, self.window_size, self.image_pair_name

    def get_tiff_metadata(self):
        """ Get the tiff metadata, which are not at the right format to be yield by the dataloader """
        return self.crs_info, self.transform_info

    def __getitem__(self, idx):
        mask = self.pre_posts[idx][0] < 10
        sample = {
            'pre_post_image': torch.tensor(self.pre_posts[idx].astype(np.float32)),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['frame_id'] = torch.tensor(idx)
        sample['x_position'] = self.x_positions[idx]
        sample['y_position'] = self.y_positions[idx]
        sample['nan_mask'] = torch.from_numpy(mask)

        return sample

