import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

image_folder = "PoC/results_500m"
output_csv = "labeled_images_500m.csv"
valid_keys = {'a', 'b', 'c', 'd', 'n'}  # 'n' for no data
label_map = {'a': 'A (Very Efficient)', 'b': 'B (Efficient)', 'c': 'C (Partially Efficient)', 'd': 'D (Inefficient)', 'n': 'N (No Data)'}

labeled = {}
if os.path.exists(output_csv):
    with open(output_csv, "r") as f:
        reader = csv.reader(f)
        labeled = {rows[0]: rows[1] for rows in reader}

images = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")])

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
    ax.set_title(f"{img_file} — press A, B, C, D, or N to label")
    ax.axis('off')

    label = None

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if label:
        # print(f"Labeled {img_file} as {label}")
        labeled[img_file] = label
        with open(output_csv, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([img_file, label])

print("All images labeled!")
