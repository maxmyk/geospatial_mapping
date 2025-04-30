#  crops images to contain only visible part of the image, omitting the transparent part

import os
from PIL import Image

from bounds2kml import get_images
from tqdm import tqdm


image_folders = [
    "results_pt_all",
]

for folder in tqdm(image_folders):
    images = get_images(folder, cropped=False)
    print(f"Found {len(images)} images in {folder}")
    for img in tqdm(images):
        if "_network.png" not in img:
            continue
        img_path = os.path.join(folder, img)
        try:
            with Image.open(img_path) as im:
                bbox = im.getbbox()
                if bbox:
                    img_path = img_path.replace("_network.png", "_network_cropped.png")
                    cropped_im = im.crop(bbox)
                    cropped_im.save(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
