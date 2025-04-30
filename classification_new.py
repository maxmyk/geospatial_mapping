import folium
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from tqdm import tqdm
from folium.plugins import HeatMap

random_state = 20250221
np.random.seed(random_state)

shape_factors = [f"phi_shape_factor{i}" for i in range(30)]
phi_area_bins = []
for bin_label in ["2_3", "3_4", "4_5"]:
    phi_area_bins.extend([f"phi_log_area_{bin_label}_{i}" for i in range(30)])
block_shape_counts = [f"n_blocks_by_shape_factor{i}" for i in range(30)]

dataset_columns = [
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
    "city_name",
    "label"
]

# Louf & Barthelemy (L&B)
shape_factors = [f"phi_shape_factor{i}" for i in range(30)]
phi_area_bins = [f"phi_log_area_{bin_label}_{i}" for bin_label in ["2_3", "3_4", "4_5"] for i in range(30)]
block_shape_counts = [f"n_blocks_by_shape_factor{i}" for i in range(30)]
louf_features = shape_factors + phi_area_bins + block_shape_counts

# Boeing
boeing_features = [
    "phi_orientation_order",
    "entropy_simplified",
    "entropy_weighted",
    "median_street_segment_length",
    "avg_circuity",
    "avg_node_degree",
    "p_dead_ends",
    "p_four_way"
]

# Our features (excluding unused/constant ones)
our_features = [
    "blocks_area",
    "span_lat",
    "span_lon",
    "number_of_blocks",
    "area_by_boundary"
]

# Hypotheses
hypothesis_sets = {
    # "L&B only": louf_features,
    # "Boeing only": boeing_features,
    # "L&B + Boeing": louf_features + boeing_features,
    # "L&B + Our": louf_features + our_features,
    # "Boeing + Our": boeing_features + our_features,
    "L&B + Boeing + Our": louf_features + boeing_features + our_features
}

locations = [
    [39.6,-8, 7],
    [38.71546938235265, -9.143135902540482, 11],
    [39.746930873276085, -8.817830135036544, 11],
    [41.079954292912205, -8.621561916660294, 11],
    [41.54737388404752, -8.44431463935501, 11],
    [41.17196189944749, -7.783940424102578, 11],
    [40.551593363201114, -7.275992996273259, 10],
    [39.74962431893158, -8.823383925711198, 10]
]

target_per_class = 200
portuguese_fraction = 0.5
df_features = pd.read_csv("data_pt12.csv")

weight_map = {"A": 8, "B":7, "C": 6, "D": 1} # the best so far

for i, loc in enumerate(locations):
    weighted_heatmap_data = []
    for _, row in df_features.iterrows():
        weight = weight_map.get(row["label"], 1)
        weighted_heatmap_data.append([row["latitude"], row["longitude"], weight])
    heatmap = HeatMap(
        data=weighted_heatmap_data, radius=15, blur=10, max_zoom=1, name="Weighted Heatmap"
    ) 
    m = folium.Map(location=[loc[0], loc[1]], zoom_start=loc[2])
    heatmap.add_to(folium.FeatureGroup(name="Weighted Heatmap").add_to(m))
    folium.LayerControl().add_to(m)
    m.save(f"heatmaps_labels_manual_{i}.html")

merge_map = {'A': 'AB', 'B': 'AB', 'C': 'C', 'D': 'D', 'AB': 'AB'}
df_features["label"] = df_features["label"].map(merge_map)
og_features = df_features.copy()
print(og_features["label"].value_counts())

df_A = pd.read_csv("augmentation_A.csv.csv")
df_B = pd.read_csv("augmentation_B.csv.csv")
df_C = pd.read_csv("augmentation_C.csv.csv")
other_features = pd.concat([df_A, df_B, df_C], ignore_index=True)
print(other_features["label"].value_counts())

df_features["label"] = df_features["label"].map(merge_map)
# dropna
df_features = df_features.dropna(subset=["label"])
other_features["label"] = other_features["label"].map(merge_map)
print(df_features["label"].value_counts())
print(other_features["label"].value_counts())
combined_features = pd.concat([df_features, other_features], ignore_index=True)
print(combined_features["label"].value_counts())

plt.figure(figsize=(5, 4))
combined_features["label"] = pd.Categorical(combined_features["label"], categories=["AB", "C", "D"], ordered=True)
sns.countplot(data=combined_features, x="label")
plt.title("Label Distribution")
plt.savefig("label_distribution_augmented_combined.png")

df_features_selected = pd.DataFrame()
other_feature_selected = pd.DataFrame()

for label in df_features["label"].unique():
    samples_of_this_label_in_df = df_features[df_features["label"] == label]
    number_of_samples = int(len(samples_of_this_label_in_df) * portuguese_fraction)
    if number_of_samples > target_per_class:
        number_of_samples = target_per_class
    the_rest = target_per_class - number_of_samples
    print(f"Label: {label}, Number of samples: {number_of_samples}, The rest: {the_rest}")
    df_features_selected = pd.concat([df_features_selected, samples_of_this_label_in_df.sample(n=number_of_samples, random_state=random_state)])
    other_feature_selected = pd.concat([other_feature_selected, other_features[other_features["label"] == label].sample(n=the_rest, random_state=random_state)])

df_features = pd.concat([df_features_selected, other_feature_selected], ignore_index=True)

df_features = df_features.dropna(subset=["label"])
df_features = df_features[df_features["zoom"] == 12]
df_features.dropna()

print(df_features["label"].value_counts())

print(len(og_features))
df_features = df_features.groupby('label').apply(lambda x: x.sample(n=target_per_class, random_state=random_state)).reset_index(drop=True)
og_features = og_features[~og_features["city_name"].isin(df_features["city_name"])]
print(len(og_features))

print(len(og_features), len(df_features))


drop_columns = ['label', 'city_name', 'latitude', 'longitude', 'zoom']
results = {}

for name, features in hypothesis_sets.items():
    if all(f in df_features.columns for f in features):
        X = df_features.drop(columns=drop_columns, errors='ignore')
        y = df_features['label'] 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=random_state
        )

        pipeline = make_pipeline(
            MinMaxScaler(),
            SelectKBest(score_func=mutual_info_classif, k=min(15, len(features))),
            SVC(kernel='rbf', class_weight='balanced', random_state=random_state)
        )
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')

        print(len(X))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))


        print("Inference (on different labeled data): ")
        la_lo = og_features[["latitude", "longitude"]].copy()
        X_og = og_features.drop(columns=drop_columns, errors='ignore')
        y_og = og_features["label"]
        y_pred = pipeline.predict(X_og)
        
        print(confusion_matrix(y_og, y_pred))
        print(classification_report(y_og, y_pred))
        df_og = og_features[["city_name", "label"]].copy()
        df_og["label_pred"] = y_pred
        df_og["latitude"] = la_lo["latitude"]
        df_og["longitude"] = la_lo["longitude"]
        df_og.to_csv(f"predictions_{name}.csv", index=False)

        for i, loc in enumerate(locations):
            m = folium.Map(location=[loc[0], loc[1]], zoom_start=loc[2])
            heatmaps = {}
            for c in df_og["label_pred"].unique():
                df_d = df_og[df_og["label_pred"] == c]
                heatmaps[c] = HeatMap(
                    data=df_d[["latitude", "longitude"]].values, radius=15, blur=10, max_zoom=1
                )
                heatmaps[c].add_to(folium.FeatureGroup(name=c, show=False).add_to(m))

            weighted_heatmap_data = []
            weight_map = {"AB": 8, "C": 6, "D": 1}
            for _, row in df_og.iterrows():
                weight = weight_map.get(row["label_pred"], 1)
                weighted_heatmap_data.append([row["latitude"], row["longitude"], weight])
            heatmap = HeatMap(
                data=weighted_heatmap_data, radius=15, blur=10, max_zoom=1, name="Weighted Heatmap"
            )
            heatmap.add_to(folium.FeatureGroup(name="Weighted Heatmap").add_to(m))

            folium.LayerControl().add_to(m)
            m.save(f"heatmaps_labels_inference_{i}.html")

        results[name] = {
            "mean_f1": scores.mean(),
            "std_f1": scores.std()
        }
    else:
        results[name] = "[ERROR] Missing features in dataset"

for name, result in results.items():
    print(f"{name}: {result}")

