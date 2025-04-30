import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from tqdm import tqdm

image_folder = "results_pt_all"
zoom = 12
output_csv = f"l_pt_{zoom}.csv"
features_csv = "results_pt_all_combined_new.csv"


valid_keys = {'a', 'b', 'c', 'd'}
label_map = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}

labeled = {}
if os.path.exists(output_csv):
    with open(output_csv, "r") as f:
        reader = csv.reader(f)
        labeled = {rows[0]+"_network.png": rows[1] for rows in reader}

images = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")])
print(len(images))

images_with_data = sorted([f.split("_")[0]+"_network.png" for f in os.listdir(image_folder) if f.endswith(".csv")])
print(len(images_with_data))
images = sorted(set(images) & set(images_with_data))
print(len(images))
images = sorted(set(images) - set(labeled.keys()))
print(len(images))

features = pd.read_csv(features_csv, index_col=-1).T.to_dict()

# selecting images by zoom
images = [img for img in images if features[int(img.split("_")[0])]["zoom"] == zoom]

# shuffle the images
import random
random.shuffle(images)

label = None
def on_key(event):
    global label
    key = event.key.lower()
    if key in valid_keys:
        label = label_map[key]
        plt.close()

for img_file in tqdm(images):
    if img_file in labeled:
        continue

    img_path = os.path.join(image_folder, img_file)
    img = mpimg.imread(img_path)

    # Show image
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"{img_file} — press A, B, C, or D to label")
    ax.axis('off')

    label = None

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if label:
        # print(f"Labeled {img_file} as {label}")
        labeled[img_file] = label
        with open(output_csv, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([img_file.split("_")[0], label])

print("All images labeled!")
