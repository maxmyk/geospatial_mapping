import pandas as pd
from scipy.stats import mode

df1 = pd.read_csv("l_images_500m.csv")
df2 = pd.read_csv("l_images_500m1.csv")
df3 = pd.read_csv("l_images_500m2.csv")

df1.columns = ["id", "label"]
df2.columns = ["id", "label"]
df3.columns = ["id", "label"]

df1["label"] = df1["label"].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4})
df2["label"] = df2["label"].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4})
df3["label"] = df3["label"].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4})

merged = df1.merge(df2, on="id", suffixes=('_1', '_2')).merge(df3, on="id")
merged.rename(columns={"label": "label_3"}, inplace=True)

label_cols = ['label_1', 'label_2', 'label_3']
labels = merged[label_cols].values

merged["mode_label"] = mode(labels, axis=1).mode.flatten()

merged["disagreement"] = merged[label_cols].nunique(axis=1) > 1

merged[["id", "label_1", "label_2", "label_3", "mode_label", "disagreement"]].to_csv("labels_with_mode.csv", index=False)

out = merged[["id", "mode_label"]].replace({1: 'A', 2: 'B', 3: 'C', 4: 'D'})
out.to_csv("labels_mode.csv", index=False)