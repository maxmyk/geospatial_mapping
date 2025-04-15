# Experiments on classification

import csv
import glob
import os
import joblib
import folium

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from folium.plugins import HeatMap

random_state = 20250415

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
df["label"] = pd.Categorical(df["label"], categories=["A", "B", "C", "D"], ordered=True)
sns.countplot(data=df, x="label")
plt.title("Label Distribution")
plt.savefig("label_distribution.png")
# plt.show()

df.to_csv("features_scaled_labeled_500m.csv", index=False)

features = df.drop(columns=["label", "city_name", "latitude", "longitude"])
X_scaled = StandardScaler().fit_transform(features)

X = features
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=random_state
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


smote = SMOTE(random_state=random_state)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_balanced.value_counts().to_dict())

nb = GaussianNB()
nb.fit(X_train_balanced, y_train_balanced)
y_pred_nb = nb.predict(X_test_scaled)
print("\n[Naive Bayes]")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


svm = SVC(kernel="rbf", C=1, gamma="scale")
# svm = SVC(kernel='rbf', class_weight='balanced', C=1, gamma='scale')
# svm.fit(X_train_scaled, y_train)
svm.fit(X_train_balanced, y_train_balanced)
y_pred_svm = svm.predict(X_test_scaled)
print("\n[SVM]")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


scores_nb = cross_val_score(nb, X_scaled, y, cv=5)
scores_svm = cross_val_score(svm, X_scaled, y, cv=5)

print(f"\n[CV Accuracy] Naive Bayes: {scores_nb.mean():.3f} ± {scores_nb.std():.3f}")
print(f"[CV Accuracy] SVM:         {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")


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


# =================================
vt = VarianceThreshold(threshold=0.0)
X_noconst = vt.fit_transform(X)
feature_names = X.columns[vt.get_support()]
X_noconst_df = pd.DataFrame(X_noconst, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_noconst_df, y, stratify=y, test_size=0.2, random_state=random_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=random_state)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

selector = SelectKBest(score_func=f_classif, k=30)
X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
X_test_selected = selector.transform(X_test_scaled)

selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X_noconst_df.columns[selected_feature_indices]
print("Top 30 features:", list(selected_feature_names))


# Raw F-scores and p-values
scores = selector.scores_
feature_scores = pd.Series(scores, index=X_noconst_df.columns)

# Top 30 features
top_features = feature_scores.loc[selected_feature_names].sort_values(ascending=True)


plt.figure(figsize=(10, 8))
top_features.plot(kind="barh")
plt.xlabel("F-Score")
plt.title("Top 30 Features Selected by SelectKBest (f_classif)")
plt.tight_layout()
plt.grid(True, axis="x", linestyle="--", alpha=0.6)
# plt.show()
plt.savefig("top_30_features.png")


rf = RandomForestClassifier(n_estimators=200, random_state=random_state)

rf.fit(X_train_selected, y_train_balanced)
y_pred_rf = rf.predict(X_test_selected)


print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[-10:]
plt.barh(np.array(selected_feature_names)[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top 10 Most Important Features")
# plt.show()

joblib.dump(rf, "best_classifier_rf.joblib")
joblib.dump(selected_feature_names.tolist(), "selected_features_rf.joblib")

X_test_full = X_test.reset_index(drop=True)
y_test_full = y_test.reset_index(drop=True)
y_pred_full = pd.Series(y_pred_rf, name="predicted")
df_plot = pd.concat([X_test_full, y_test_full, y_pred_full], axis=1)
df_plot["correct"] = df_plot["label"] == df_plot["predicted"]
m = folium.Map(location=[0, 0], zoom_start=2)

for _, row in df_plot.iterrows():
    color = "green" if row["correct"] else "red"
    popup_text = f"City: {df.iloc[_]['city_name']}<br>True: {row['label']}<br>Predicted: {row['predicted']}"
    la = df.iloc[_]["latitude"]
    lo = df.iloc[_]["longitude"]
    folium.CircleMarker(
        location=[la, lo],
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(popup_text, max_width=300),
    ).add_to(m)
m.save("misclassifications_map.html")


# =================================


param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=random_state),
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
)
grid_search.fit(X_train_selected, y_train_balanced)

print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)


le = LabelEncoder()
y_train_balanced_enc = le.fit_transform(y_train_balanced)
y_test_enc = le.transform(y_test)

xgb = XGBClassifier(
    random_state=random_state, use_label_encoder=False, eval_metric="mlogloss"
)
xgb.fit(X_train_selected, y_train_balanced_enc)
y_pred_enc = xgb.predict(X_test_selected)
y_pred = le.inverse_transform(y_pred_enc)


svm = SVC(kernel="rbf", probability=True, C=1, gamma="scale")

voting = VotingClassifier(
    estimators=[("svm", svm), ("rf", rf), ("xgb", xgb)],
    voting="soft",  # average predicted probabilities
)
voting.fit(X_train_selected, y_train_balanced)

print("\n[Voting Classifier]")
y_pred_voting = voting.predict(X_test_selected)
print(confusion_matrix(y_test, y_pred_voting))
print(classification_report(y_test, y_pred_voting))


m = folium.Map(location=[0, 0], zoom_start=2)
heatmaps = {}
for c in df["label"].unique():
    df_d = df[df["label"] == c]
    heatmaps[c] = HeatMap(
        data=df_d[["latitude", "longitude"]].values, radius=15, blur=10, max_zoom=1
    )
    heatmaps[c].add_to(folium.FeatureGroup(name=c).add_to(m))

folium.LayerControl().add_to(m)
m.save("heatmaps_all.html")


# More experiments


csv_files = glob.glob("PoC/results_500m_4k/*.csv")
city_names = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f in csv_files]
df_list = []
for f, name in zip(csv_files, city_names):
    row = pd.read_csv(f, header=None)
    row["city_name"] = name
    df_list.append(row)
df_unlabeled = pd.concat(df_list, ignore_index=True)
data = df_unlabeled
data.columns = feature_columns
data = data.drop(columns=["latitude", "longitude", "city_name"])
data = data[data["blocks_area"] > 1]
data = data[data["blocks_area"] < 10]
df_unlabeled = df_unlabeled.dropna()
X_unlabeled = df_unlabeled[selected_feature_names]

print("X_train_selected columns:", X_train_selected.shape)
print("X_unlabeled_scaled columns:", X_unlabeled.shape)

scaler = StandardScaler()
X_unlabeled_scaled = scaler.fit_transform(X_unlabeled)
y_unlabeled_pred_enc = voting.predict(X_unlabeled_scaled)
df_unlabeled["predicted_label"] = y_unlabeled_pred_enc


plt.figure(figsize=(6, 4))
sns.countplot(data=df_unlabeled, x="predicted_label", order=["A", "B", "C", "D"])
plt.title("Predicted Class Distribution for Unlabeled Cities")
plt.xlabel("Predicted Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("predicted_class_distribution.png")
# plt.show()

map_center = [df_unlabeled["latitude"].mean(), df_unlabeled["longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=5)

color_map = {"A": "red", "B": "blue", "C": "green", "D": "purple"}

for _, row in df_unlabeled.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=4,
        color=color_map.get(row["predicted_label"], "gray"),
        fill=True,
        fill_opacity=0.6,
        popup=f"City: {row['city_name']}<br>Predicted: {row['predicted_label']}",
    ).add_to(m)

m.save("unlabeled_predictions_map.html")

m = folium.Map(location=[0, 0], zoom_start=2)
heatmaps = {}
for c in df_unlabeled["predicted_label"].unique():
    df_d = df_unlabeled[df_unlabeled["predicted_label"] == c]
    heatmaps[c] = HeatMap(
        data=df_d[["latitude", "longitude"]].values, radius=15, blur=10, max_zoom=1
    )
    heatmaps[c].add_to(folium.FeatureGroup(name=c, show=False).add_to(m))

weighted_heatmap_data = []
weight_map = {"A": 8, "B": 4, "C": 2, "D": 1}
for _, row in df_unlabeled.iterrows():
    weight = weight_map.get(row["predicted_label"], 1)
    weighted_heatmap_data.append([row["latitude"], row["longitude"], weight])
heatmap = HeatMap(
    data=weighted_heatmap_data, radius=15, blur=10, max_zoom=1, name="Weighted Heatmap"
)
heatmap.add_to(folium.FeatureGroup(name="Weighted Heatmap").add_to(m))

folium.LayerControl().add_to(m)
m.save("heatmaps_all_predictions.html")
