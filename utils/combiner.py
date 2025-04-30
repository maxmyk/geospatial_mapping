import csv
import glob
import os

import pandas as pd

features_csv = "results_pt_all/all_features.csv"
labels_csv =   "l_pt_12.csv"
output_csv =   "results_pt_all_combined.csv"
zoom = 12

if not os.path.exists(features_csv):
    raise FileNotFoundError(f"Features file {features_csv} does not exist.")
if not os.path.exists(labels_csv):
    raise FileNotFoundError(f"Labels file {labels_csv} does not exist.")



df = pd.read_csv(features_csv)
df = df[df.zoom == zoom]

labels = {}
with open(labels_csv, "r") as f:
    reader = csv.reader(f)
    for rows in reader:
        labels[int(rows[0])] = rows[1]
df["label"] = df.city_name.map(labels)
df = df.dropna(subset=["label"])

df.to_csv(output_csv, index=False)
print(f"Saved combined data to {output_csv}")