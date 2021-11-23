import os
import glob
import json

from PIL import Image
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
from torch.utils.data import Dataset
from imgaug.augmentables import Keypoint
from imgaug.augmentables import KeypointsOnImage

import numpy as np
import imgaug.augmenters as iaa


class LandmarkDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, cfg_dataset, perform_augmentation=False):

        self.cfg_dataset = cfg_dataset
        self.perform_augmentation = perform_augmentation

        # Define augmentation
        data_aug_params = cfg_dataset.AUGMENTATION
        self.augmentation = iaa.Sequential([
            iaa.Affine(translate_px={"x": (-data_aug_params.TRANSLATION_X, data_aug_params.TRANSLATION_X),
                                     "y": (-data_aug_params.TRANSLATION_Y, data_aug_params.TRANSLATION_Y)},
                       scale=[1 - data_aug_params.SF, 1],
                       rotate=[-data_aug_params.ROTATION_FACTOR, data_aug_params.ROTATION_FACTOR]),
            iaa.Multiply(mul=(1 - data_aug_params.INTENSITY_FACTOR, 1 + data_aug_params.INTENSITY_FACTOR)),
            iaa.GammaContrast(),
            iaa.ElasticTransformation(alpha=(0, data_aug_params.ELASTIC_STRENGTH),
                                      sigma=data_aug_params.ELASTIC_SMOOTHNESS, order=3)
        ])

        self.db = self.cache(image_dir, annotation_dir, cfg_dataset)


    @staticmethod
    def cache(images_dir, annotation_dir, cfg_dataset):

        db = []

        # Rename these parameters for clarity
        downsampled_image_width = cfg_dataset.CACHED_IMAGE_SIZE[0]
        downsampled_image_height = cfg_dataset.CACHED_IMAGE_SIZE[1]

        # Aspect ratio is defined as width / height
        downsampled_aspect_ratio = downsampled_image_width / downsampled_image_height

        # Define how to downsample and pad images
        preprocessing_steps = [
            iaa.PadToAspectRatio(downsampled_aspect_ratio, position='right-bottom'),
            iaa.Resize({"width": downsampled_image_width, "height": downsampled_image_height}),
        ]
        seq = iaa.Sequential(preprocessing_steps)

        # We save the cached images in the config CACHE_DIR/ImageWidth_ImageHeight
        cache_data_dir = os.path.join(cfg_dataset.CACHE_DIR, "{}_{}".format(downsampled_image_width, downsampled_image_height))
        if not os.path.exists(cache_data_dir):
            os.makedirs(cache_data_dir)

        # get the file names of all images in the directory
        image_paths = sorted(glob.glob(images_dir + "/*" + cfg_dataset.IMAGE_EXT))

        for image_path in tqdm(image_paths):

            # Get the file name with no extension
            file_name = os.path.basename(image_path).split(".")[0]

            # Get sub-directories for annotations
            annotation_sub_dirs = sorted(glob.glob(annotation_dir + "/*"))

            # Keep track of where we will be saving the downsampled image and the meta data
            cache_image_path = os.path.join(cache_data_dir, file_name + ".png")
            cache_meta_path = os.path.join(cache_data_dir, file_name + "_meta.txt")
            cache_annotation_paths = []

            annotation_paths = []
            for annotation_sub_dir in annotation_sub_dirs:
                annotation_paths.append(os.path.join(annotation_sub_dir, file_name + ".txt"))
                sub_dir_name = annotation_sub_dir.split("/")[-1]
                cache_annotation_paths.append(os.path.join(cache_data_dir, file_name + "_" + sub_dir_name + ".txt"))

            db.append({
                "cached_image_path": cache_image_path,
                "cached_annotation_paths": cache_annotation_paths,
                "cached_meta_path": cache_meta_path
            })

            # Don't need to create them if they already exist
            if not os.path.exists(cache_image_path) or \
                    not os.path.exists(cache_meta_path):

                # -----------Image-----------

                # Get image
                image = io.imread(image_path, as_gray=True)

                # Augment image
                image_resized = seq(image=image)

                # Save new image
                image_resized = np.clip(image_resized, 0.0, 1.0)
                image_as_255 = img_as_ubyte(image_resized)
                im = Image.fromarray(image_as_255)
                im.save(cache_image_path)

                # -----------Annotations-----------

                # Use pandas to extract the key points from the txt file
                for annotation_path, cache_annotation_path in zip(annotation_paths, cache_annotation_paths):

                    # Get annotations
                    kps_np_array = np.loadtxt(annotation_path, delimiter=",", max_rows=cfg_dataset.KEY_POINTS)

                    # Augment annotations
                    kps = KeypointsOnImage.from_xy_array(kps_np_array, shape=image.shape)
                    kps_resized = seq(keypoints=kps)

                    # Save annotations
                    kps_np_array = kps_resized.to_xy_array()
                    np.savetxt(cache_annotation_path, kps_np_array, fmt="%.14g", delimiter=" ")

                # -----------Meta Data-----------

                # image.shape is rows x columns
                original_image_height, original_image_width = image.shape
                original_aspect_ratio = original_image_width / original_image_height

                if original_aspect_ratio > downsampled_aspect_ratio:
                    scale_factor = original_image_width / downsampled_image_width
                else:
                    scale_factor = original_image_height / downsampled_image_height

                # Get pixel size
                meta_dict = {
                    "file_name": file_name,
                    "scale_factor": scale_factor
                }
                with open(cache_meta_path, 'w') as file:
                    file.write(json.dumps(meta_dict))

        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):

        cached_image_path = self.db[idx]["cached_image_path"]
        cached_annotation_paths = self.db[idx]["cached_annotation_paths"]
        cached_meta_path = self.db[idx]["cached_meta_path"]

        # get image
        image = io.imread(cached_image_path, as_gray=True)

        # Use pandas to extract the key points from the txt file
        landmarks_per_annotator = []

        for cached_annotation_path in cached_annotation_paths:
            kps_np_array = np.loadtxt(cached_annotation_path, delimiter=" ")
            landmarks_per_annotator.append(kps_np_array)

        if self.perform_augmentation:

            # Augment image and annotations at the same to ensure the augmentation is the same
            kps = KeypointsOnImage.from_xy_array(np.concatenate(landmarks_per_annotator), shape=image.shape)
            image, kps_augmented = self.augmentation(image=image, keypoints=kps)
            landmarks_per_annotator = kps_augmented.to_xy_array().reshape(-1, self.cfg_dataset.KEY_POINTS, 2)

        # This line is here so we slice the array later in the code
        landmarks_per_annotator = np.array(landmarks_per_annotator)

        # Generate ground truth maps
        channels = np.zeros([self.cfg_dataset.KEY_POINTS, image.shape[0], image.shape[1]])
        for i in range(self.cfg_dataset.KEY_POINTS):
            x, y = np.mean(landmarks_per_annotator[:, i], axis=0).astype(int)
            channels[i, y, x] = 1.0

        # Add a channel to the images
        image = np.expand_dims(image, axis=0)

        # Get meta
        meta_file = open(cached_meta_path, "r")

        meta = json.load(meta_file)
        meta["landmarks_per_annotator"] = landmarks_per_annotator.copy()
        meta["pixel_size"] = np.array(self.cfg_dataset.PIXEL_SIZE) * meta["scale_factor"]
        return image, channels, meta