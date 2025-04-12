# Experiments on classification

import csv
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# from sklearn.decomposition import PCA

shape_factors = [f"phi_shape_factor{i}" for i in range(30)]
phi_area_bins = []
for bin_label in ["2_3", "3_4", "4_5"]:
    phi_area_bins.extend([f"phi_log_area_{bin_label}_{i}" for i in range(30)])
block_shape_counts = [f"n_blocks_by_shape_factor{i}" for i in range(30)]

feature_columns = [
    "phi_orientation_order",
    "entropy_simplified",
    "entropy_weighted",
    "median_street_segment_length",
    "avg_circuity",
    "avg_node_degree",
    "p_dead_ends",
    "p_four_way",
    *shape_factors,
    *phi_area_bins,
    "zoom",
    "blocks_area",
    "span_lat",
    "span_lon",
    "number_of_blocks",
    *block_shape_counts,
    "area_by_boundary",
    "latitude",
    "longitude",
    "zoom_lat",
    "zoom_lon",
]

csv_files = glob.glob("PoC/results_500m/*.csv")
city_names = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f in csv_files]
df_list = []
for f, name in zip(csv_files, city_names):
    row = pd.read_csv(f, header=None)
    row["city_name"] = name
    df_list.append(row)
df = pd.concat(df_list, ignore_index=True)

data = df
feature_columns.append("city_name")
data.columns = feature_columns
data = data.drop(columns=["latitude", "longitude", "city_name"])

# making sure the area is in correct range, should be changed for different are bins
data = data[data["blocks_area"] > 1]
data = data[data["blocks_area"] < 10]

labels = {}
with open("labeled_images_500m_1.csv", "r") as f:
    reader = csv.reader(f)
    for rows in reader:
        labels[rows[0]] = rows[1]
df["label"] = df["city_name"].map(labels)
df = df.dropna(subset=["label"])

print(df.head())
print(df["label"].value_counts())

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="label")
plt.title("Label Distribution")
plt.savefig("label_distribution.png")
# plt.show()


features = df.drop(columns=["label", "city_name"])
X_scaled = StandardScaler().fit_transform(features)

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# plt.figure(figsize=(6, 5))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["label"])
# plt.title("PCA Projection of Features")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()


X = features
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=20250412
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print("\n[Naive Bayes]")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
print("\n[SVM]")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


scores_nb = cross_val_score(nb, X_scaled, y, cv=5)
scores_svm = cross_val_score(svm, X_scaled, y, cv=5)

print(f"\n[CV Accuracy] Naive Bayes: {scores_nb.mean():.3f} ± {scores_nb.std():.3f}")
print(f"[CV Accuracy] SVM:         {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")


# Plottinh
cm = confusion_matrix(y_test, y_pred_svm, labels=["A", "B", "C", "D"])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["A", "B", "C", "D"],
    yticklabels=["A", "B", "C", "D"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
# plt.show()


# # Folium experiments
# import pandas as pd
# import folium
# from folium.plugins import MarkerCluster

# m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
# marker_cluster = MarkerCluster().add_to(m)

# color_map = {
#     'A': 'red',
#     'B': 'blue',
#     'C': 'green',
#     'D': 'purple'
# }

# for _, row in df.iterrows():
#     folium.CircleMarker(
#         location=(row['latitude'], row['longitude']),
#         radius=5,
#         color=color_map.get(row['label'], 'black'),
#         fill=True,
#         fill_opacity=0.7,
#         popup=f"{row['city_name']} - {row['label']}"
#     ).add_to(marker_cluster)

# m.save('city_label_heatmap.html')

# Not the best variant, but works
# import geopandas as gpd
# import matplotlib.pyplot as plt
# world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
# fig, ax = plt.subplots(figsize=(12, 8))
# world.plot(ax=ax, color='lightgrey')
# gdf.plot(ax=ax, column='label', categorical=True, legend=True, markersize=30)
# plt.title('Geographic Distribution of City Classes')
# plt.show()


from folium.plugins import HeatMap
import folium

for c in df["label"].unique():
    df_d = df[df["label"] == c]
    m = folium.Map(location=[0, 0], zoom_start=2)
    HeatMap(
        data=df_d[["latitude", "longitude"]].values, radius=15, blur=10, max_zoom=1
    ).add_to(m)

    m.save(f"heatmap_class_{c}.html")
