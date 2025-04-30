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
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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

# csv_files = glob.glob("PoC/results_500m/*.csv")
# csv_files = glob.glob("PoC/results_500m_4k/*.csv")
csv_files = glob.glob("PoC/results_pt_all/*.csv")
# csv_files = glob.glob("PoC/results_ro_all_combined/*.csv")
city_names = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f in csv_files]
df_list = []
for f, name in zip(csv_files, city_names):
    row = pd.read_csv(f, header=None)
    row["city_name"] = name
    df_list.append(row)
df = pd.concat(df_list, ignore_index=True)
# save the data for later use


# save the data for later use

data = df
feature_columns.append("city_name")
data.columns = feature_columns
# using only zoom level 12
data = data[data["zoom"] == 12]
# data = data.drop(columns=["latitude", "longitude", "city_name"])

# plotting zoom distribution
plt.figure(figsize=(5, 4))
sns.histplot(data=data, x="zoom", bins=30, kde=True)
plt.title("Zoom Distribution")
plt.xlabel("Zoom Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("zoom_distribution.png")

# using only zoom level 12
df = df[df["zoom"] == 12]


# making sure the area is in correct range, should be changed for different are bins
# data = data[data["blocks_area"] > 1]
# data = data[data["blocks_area"] < 10]

labels = {}
# with open("labels_mode_500m.csv", "r") as f:
# with open("l_images_4km.csv", "r") as f:
with open("l_pt_12.csv", "r") as f:
# with open("l_ro.csv", "r") as f:
    reader = csv.reader(f)
    for rows in reader:
        labels[rows[0]] = rows[1]
df["label"] = df["city_name"].map(labels)
df = df.dropna(subset=["label"])

print(df.head())
print(df["label"].value_counts())

# real A from labeled data that are zoom 12
# df_A = df[df["label"] == "C"]
# df_A = df_A[df_A["zoom"] == 12]

# save the candidates to a CSV file
# df_A.to_csv("real_C_candidates_4k12.csv", index=False)
# print("Real A candidates count:", len(df_A))
# exit(0)

# adding data from other countries for A and B labels to balance the dataset
df_A = pd.read_csv("real_A_candidates.csv")
df_B = pd.read_csv("real_B_candidates.csv")
df_A1 = pd.read_csv("real_A_candidates_500m.csv")
df_B1 = pd.read_csv("real_B_candidates_500m.csv")
df_C = pd.read_csv("real_C_candidates_4k12.csv")

# extending the dataset with real A and B candidates
df = pd.concat([df, df_A, df_B, df_A1, df_B1, df_C], ignore_index=True)
df = df.dropna(subset=["label"])
df['label'] = df['label'].replace(merge_map)
print(df["label"].value_counts())

# converting to ternary classification by merging A and B (changing the label to "AB")

# df["label"] = df["label"].replace({"A": "AB", "B": "AB"})

# plt.figure(figsize=(5, 4))
# df["label"] = pd.Categorical(df["label"], categories=["A", "B", "C", "D"], ordered=True)
# sns.countplot(data=df, x="label")
# plt.title("Label Distribution")
# plt.savefig("label_distribution.png")
# # plt.show()

# # plotting zoom distribution by label
# # plt.figure(figsize=(5, 4))
# # sns.histplot(data=df, x="zoom", bins=30, kde=True, hue="label", multiple="stack")
# # plt.title("Zoom Distribution by Label")
# # plt.xlabel("Zoom Level")
# # plt.ylabel("Frequency")
# # plt.tight_layout()
# # plt.savefig("zoom_distribution_by_label.png")
# # exit(0)

# # df.to_csv("features_scaled_labeled_500m.csv", index=False)

# features = df.drop(columns=["label", "city_name", "latitude", "longitude"])
# X_scaled = StandardScaler().fit_transform(features)

# X = features
# y = df["label"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, stratify=y, test_size=0.2, random_state=random_state
# )
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# # smote = SMOTE(random_state=random_state)
# # X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# # print("Before SMOTE:", y_train.value_counts().to_dict())
# # print("After SMOTE:", y_train_balanced.value_counts().to_dict())

# # Balancing by undersampling the majority class
# from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter

# rus = RandomUnderSampler(random_state=random_state)
# X_train_balanced, y_train_balanced = rus.fit_resample(X_train_scaled, y_train)
# print("Before Undersampling:", Counter(y_train).items())
# print("After Undersampling:", Counter(y_train_balanced).items())

# nb = GaussianNB()
# nb.fit(X_train_balanced, y_train_balanced)
# y_pred_nb = nb.predict(X_test_scaled)
# print("\n[Naive Bayes]")
# print(confusion_matrix(y_test, y_pred_nb))
# print(classification_report(y_test, y_pred_nb))


# svm = SVC(kernel="rbf", C=1, gamma="scale")
# # svm = SVC(kernel='rbf', class_weight='balanced', C=1, gamma='scale')
# # svm.fit(X_train_scaled, y_train)
# svm.fit(X_train_balanced, y_train_balanced)
# y_pred_svm = svm.predict(X_test_scaled)
# print("\n[SVM]")
# print(confusion_matrix(y_test, y_pred_svm))
# print(classification_report(y_test, y_pred_svm))


# scores_nb = cross_val_score(nb, X_scaled, y, cv=5)
# scores_svm = cross_val_score(svm, X_scaled, y, cv=5)

# print(f"\n[CV Accuracy] Naive Bayes: {scores_nb.mean():.3f} ± {scores_nb.std():.3f}")
# print(f"[CV Accuracy] SVM:         {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")


# # plt.show()


# # =================================
# vt = VarianceThreshold(threshold=0.0)
# X_noconst = vt.fit_transform(X)
# feature_names = X.columns[vt.get_support()]
# X_noconst_df = pd.DataFrame(X_noconst, columns=feature_names)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_noconst_df, y, stratify=y, test_size=0.2, random_state=random_state
# )

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # SMOTE
# # smote = SMOTE(random_state=random_state)
# # X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# # # Undersampling
# rus = RandomUnderSampler(random_state=random_state)
# X_train_balanced, y_train_balanced = rus.fit_resample(X_train_scaled, y_train)


# selector = SelectKBest(score_func=f_classif, k=30)
# X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
# X_test_selected = selector.transform(X_test_scaled)

# selected_feature_indices = selector.get_support(indices=True)
# selected_feature_names = X_noconst_df.columns[selected_feature_indices]
# print("Top 30 features:", list(selected_feature_names))


# # Raw F-scores and p-values
# scores = selector.scores_
# feature_scores = pd.Series(scores, index=X_noconst_df.columns)

# # Top 30 features
# top_features = feature_scores.loc[selected_feature_names].sort_values(ascending=True)


# plt.figure(figsize=(10, 8))
# top_features.plot(kind="barh")
# plt.xlabel("F-Score")
# plt.title("Top 30 Features Selected by SelectKBest (f_classif)")
# plt.tight_layout()
# plt.grid(True, axis="x", linestyle="--", alpha=0.6)
# # plt.show()
# plt.savefig("top_30_features.png")


# rf = RandomForestClassifier(n_estimators=200, random_state=random_state)

# rf.fit(X_train_selected, y_train_balanced)
# y_pred_rf = rf.predict(X_test_selected)


# print(confusion_matrix(y_test, y_pred_rf))
# print(classification_report(y_test, y_pred_rf))

# importances = rf.feature_importances_
# sorted_idx = np.argsort(importances)[-10:]
# plt.barh(np.array(selected_feature_names)[sorted_idx], importances[sorted_idx])
# plt.xlabel("Feature Importance")
# plt.title("Top 10 Most Important Features")
# # plt.show()

# joblib.dump(rf, "best_classifier_rf.joblib")
# joblib.dump(selected_feature_names.tolist(), "selected_features_rf.joblib")

# X_test_full = X_test.reset_index(drop=True)
# y_test_full = y_test.reset_index(drop=True)
# y_pred_full = pd.Series(y_pred_rf, name="predicted")
# df_plot = pd.concat([X_test_full, y_test_full, y_pred_full], axis=1)
# df_plot["correct"] = df_plot["label"] == df_plot["predicted"]
# m = folium.Map(location=[0, 0], zoom_start=2)

# for _, row in df_plot.iterrows():
#     color = "green" if row["correct"] else "red"
#     popup_text = f"City: {df.iloc[_]['city_name']}<br>True: {row['label']}<br>Predicted: {row['predicted']}"
#     la = df.iloc[_]["latitude"]
#     lo = df.iloc[_]["longitude"]
#     folium.CircleMarker(
#         location=[la, lo],
#         radius=6,
#         color=color,
#         fill=True,
#         fill_opacity=0.8,
#         popup=folium.Popup(popup_text, max_width=300),
#     ).add_to(m)
# m.save("misclassifications_map.html")


# # =================================


# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [5, 10, 20, None],
#     "min_samples_split": [2, 5, 10],
# }

# grid_search = GridSearchCV(
#     RandomForestClassifier(random_state=random_state),
#     param_grid,
#     cv=5,
#     scoring="f1_weighted",
#     n_jobs=-1,
# )
# grid_search.fit(X_train_selected, y_train_balanced)

# print("Best params:", grid_search.best_params_)
# print("Best CV score:", grid_search.best_score_)


# le = LabelEncoder()
# y_train_balanced_enc = le.fit_transform(y_train_balanced)
# y_test_enc = le.transform(y_test)

# xgb = XGBClassifier(
#     random_state=random_state, use_label_encoder=False, eval_metric="mlogloss"
# )
# xgb.fit(X_train_selected, y_train_balanced_enc)
# y_pred_enc = xgb.predict(X_test_selected)
# y_pred = le.inverse_transform(y_pred_enc)


# svm = SVC(kernel="rbf", probability=True, C=1, gamma="scale")

# voting = VotingClassifier(
#     estimators=[("svm", svm), ("rf", rf), ("xgb", xgb)],
#     voting="soft",  # average predicted probabilities
# )
# voting.fit(X_train_selected, y_train_balanced)

# print("\n[Voting Classifier]")
# y_pred_voting = voting.predict(X_test_selected)
# print(confusion_matrix(y_test, y_pred_voting))
# print(classification_report(y_test, y_pred_voting))

# cm = confusion_matrix(y_test, y_pred_svm, labels=["A", "B", "C", "D"])

# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["A", "B", "C", "D"],
#     yticklabels=["A", "B", "C", "D"],
# )
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix Heatmap for SVM")
# plt.tight_layout()
# plt.savefig("confusion_matrix_svm.png")

# # center = average latitude and longitude
# map_center = [df["latitude"].mean(), df["longitude"].mean()]
# m = folium.Map(location=map_center, zoom_start=6)
# heatmaps = {}
# for c in df["label"].unique():
#     df_d = df[df["label"] == c]
#     heatmaps[c] = HeatMap(
#         data=df_d[["latitude", "longitude"]].values, radius=15, blur=10, max_zoom=1
#     )
#     heatmaps[c].add_to(folium.FeatureGroup(name=c).add_to(m))

# folium.LayerControl().add_to(m)
# m.save("heatmaps_all.html")


# # More experiments


# # csv_files = glob.glob("PoC/results_m_drive_alt/*.csv")
# # csv_files = glob.glob("PoC/results_500m_4k/*.csv")
csv_files = glob.glob("PoC/results_pt_all/*.csv")
# # csv_files = glob.glob("PoC/results_ro_all_combined/*.csv")
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
# data = data[data["blocks_area"] > 1]
# data = data[data["blocks_area"] < 10]
df_unlabeled = df_unlabeled.dropna()
df_unlabeled = df_unlabeled[df_unlabeled["zoom"] == 12]
X_unlabeled = df_unlabeled.drop(columns=['label', 'city_name', 'latitude', 'longitude', 'zoom_lat', 'zoom_lon', 'zoom'], errors='ignore')

# print("X_train_selected columns:", X_train_selected.shape)
# print("X_unlabeled_scaled columns:", X_unlabeled.shape)

# scaler = StandardScaler()
# X_unlabeled_scaled = scaler.fit_transform(X_unlabeled)
# y_unlabeled_pred_enc = voting.predict(X_unlabeled_scaled)
# df_unlabeled["predicted_label"] = y_unlabeled_pred_enc

# using default SVM with 
# merge_strategy	sampler	scaler	class_weight	kernel	best_params
# AB	none	maxabs	manual_fibonacci	rbf	{'svm__C': 10, 'svm__gamma': 'scale'}

# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MaxAbsScaler
from imblearn.pipeline import Pipeline as ImbPipeline
# from sklearn.metrics import f1_score
# import pandas as pd

# ========== CONFIGURATION ==========
merge_map = {'A': 'AB', 'B': 'AB', 'C': 'C', 'D': 'D'}
scaler = MinMaxScaler()
class_weight = {'A': 3, 'B': 2, 'C': 1, 'D': 1}
kernel = 'rbf'
best_params = {'C': 1, 'gamma': 'scale'}
# sampler = None  # no sampling
# using undersampling
from imblearn.under_sampling import RandomUnderSampler
sampler = None#RandomUnderSampler(random_state=random_state)

# manually dropping the number of samples to 200 for class AB, 200 for C, 200 for D
df = df.groupby('label').apply(lambda x: x.sample(n=200, random_state=random_state)).reset_index(drop=True)

# ========== APPLY MERGE STRATEGY ==========
y = df['label'].replace(merge_map)
X = df.drop(columns=['label', 'city_name', 'latitude', 'longitude', 'zoom_lat', 'zoom_lon', 'zoom'], errors='ignore')


# # =======================================================================================
from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# import pandas as pd
# import numpy as np

# # Example setup — replace with your actual data
# X = df.drop(columns=["label", "city_name", "latitude", "longitude"], errors='ignore')
# y = df["label"]

# # Normalize features
# scaler = MaxAbsScaler

# # === 1. SelectKBest: Top 30 features ===
# k = 30
# selector = SelectKBest(score_func=mutual_info_classif, k=k)
# clf_kbest = make_pipeline(scaler, selector, SVC(kernel='rbf', class_weight='balanced'))

# # Cross-validation score
# scores_kbest = cross_val_score(clf_kbest, X, y, cv=5, scoring='f1_weighted')
# print(f"SelectKBest (top {k} features) — Weighted F1: {scores_kbest.mean():.3f} ± {scores_kbest.std():.3f}")

# # === 2. PCA: Reduce to 30 components ===
# n_components = 30
# pca = PCA(n_components=n_components)
# clf_pca = make_pipeline(scaler, pca, SVC(kernel='rbf', class_weight='balanced'))

# scores_pca = cross_val_score(clf_pca, X, y, cv=5, scoring='f1_weighted')
# print(f"PCA ({n_components} components) — Weighted F1: {scores_pca.mean():.3f} ± {scores_pca.std():.3f}")



# ks = [5, 10, 15, 20, 30, 40, 50, 60]
# scores_kbest = []

# for k in ks:
#     selector = SelectKBest(score_func=mutual_info_classif, k=k)
#     clf_k = make_pipeline(scaler, selector, SVC(kernel='rbf', class_weight='balanced'))
#     scores = cross_val_score(clf_k, X, y, cv=5, scoring='f1_weighted')
#     scores_kbest.append(scores.mean())  # Not the full scores array!

# print(f"SelectKBest (top {k} features) — Weighted F1: {scores_kbest[-1]:.3f} ± {np.std(scores_kbest):.3f}")
# print(scores, scores_kbest)
# # Plot correctly
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(ks, scores_kbest, marker='o')
# plt.ylim(0, 1)  # F1-score is always between 0 and 1
# plt.xlabel("Number of Features (SelectKBest)")
# plt.ylabel("Weighted F1 Score")
# plt.title("Model Performance vs Feature Count")
# plt.grid(True)
# # plt.show()

selector = SelectKBest(score_func=mutual_info_classif, k=15)
selector.fit(X, y)

# # Get boolean mask of selected features
mask = selector.get_support()

# # Get the actual feature names
selected_features = X.columns[mask].tolist()
print("Top 15 selected features:")
print(selected_features)

X = X[selected_features]


# clf_final = make_pipeline(
#     StandardScaler(),
#     SelectKBest(score_func=mutual_info_classif, k=15),
#     SVC(kernel='rbf', class_weight='balanced', probability=True)
# )

# # Train-test split (or use CV if you're evaluating)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
# # print dimensions of train and test sets
# print(f"Train size: {len(X_train[0])}, Test size: {len(X_test[0])}")

# # Fit the model
# clf_final.fit(X_train, y_train)

# # Predict on test
# y_pred = clf_final.predict(X_test)

# # metrics
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.3f}")

# exit(0)
# =======================================================================================


df_selected = df[selected_features + ['label']]


# Filter class weights to only present labels
class_weight = {cls: w for cls, w in class_weight.items() if cls in y.unique()}

# ========== SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# print test and train sizes
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ========== PIPELINE ==========
steps = []
if sampler:
    steps.append(('sampler', sampler))
steps.append(('scaler', scaler))
steps.append(('svm', SVC(kernel=kernel, class_weight='balanced', **best_params)))

pipeline = ImbPipeline(steps)

# ========== TRAIN & EVALUATE ==========
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# print

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

df_unlabeled["predicted_label"] = pipeline.predict(X_unlabeled)

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"Weighted F1: {weighted_f1:.3f}")
print(f"Macro F1: {macro_f1:.3f}")


# ================================
# probs = voting.predict_proba(X_unlabeled_scaled)
# conf = probs.max(axis=1)
# preds = voting.predict(X_unlabeled_scaled)

# high_conf_A = (preds == 'A') & (conf > 0.75)
# df_A_candidates = df_unlabeled[high_conf_A]

# print("Predicted A candidates:")
# print(df_A_candidates[["city_name", "predicted_label"]])
# # save the candidates to a CSV file
# df_A_candidates.to_csv("predicted_A_candidates_75.csv", index=False)
# print("Predicted A candidates count:", len(df_A_candidates))
# ================================

plt.figure(figsize=(6, 4))
sns.countplot(data=df_unlabeled, x="predicted_label", order=["A", "B", "C", "D"])
plt.title("Predicted Class Distribution for Unlabeled Cities")
plt.xlabel("Predicted Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("predicted_class_distribution.png")
# plt.show()

# lat max - lat min, lon max - lon min

# m = folium.Map(location=map_center, zoom_start=5)

# color_map = {"A": "red", "B": "blue", "C": "green", "D": "purple"}

# for _, row in df_unlabeled.iterrows():
#     folium.CircleMarker(
#         location=[row["latitude"], row["longitude"]],
#         radius=4,
#         color=color_map.get(row["predicted_label"], "gray"),
#         fill=True,
#         fill_opacity=0.6,
#         popup=f"City: {row['city_name']}<br>Predicted: {row['predicted_label']}",
#     ).add_to(m)

# m.save("unlabeled_predictions_map.html")

m = folium.Map(location=map_center, zoom_start=7)
heatmaps = {}
for c in df_unlabeled["predicted_label"].unique():
    df_d = df_unlabeled[df_unlabeled["predicted_label"] == c]
    heatmaps[c] = HeatMap(
        data=df_d[["latitude", "longitude"]].values, radius=15, blur=10, max_zoom=1
    )
    heatmaps[c].add_to(folium.FeatureGroup(name=c, show=False).add_to(m))

weighted_heatmap_data = []
weight_map = {"AB": 4, "C": 2, "D": 1}
for _, row in df_unlabeled.iterrows():
    weight = weight_map.get(row["predicted_label"], 1)
    weighted_heatmap_data.append([row["latitude"], row["longitude"], weight])
heatmap = HeatMap(
    data=weighted_heatmap_data, radius=15, blur=10, max_zoom=1, name="Weighted Heatmap"
)
heatmap.add_to(folium.FeatureGroup(name="Weighted Heatmap").add_to(m))

folium.LayerControl().add_to(m)
m.save("heatmaps_all_predictions_PT.html")
