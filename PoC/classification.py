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

df = pd.read_csv("results_pt_all_combined.csv")

plt.figure(figsize=(5, 4))
sns.histplot(data=df, x="zoom", bins=30, kde=True)
plt.title("Zoom Distribution")
plt.xlabel("Zoom Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("zoom_distribution.png")

# using only zoom level 12
df = df[df["zoom"] == 12]

labels = {}
with open("l_pt_12.csv", "r") as f:
    reader = csv.reader(f)
    for rows in reader:
        labels[int(rows[0])] = rows[1]
df["label"] = df["city_name"].map(labels)
df = df.dropna(subset=["label"])

print(df.head())
print(df["label"].value_counts())

# adding data from other countries for A and B labels to balance the dataset
df_A = pd.read_csv("augmentation_A.csv")
df_B = pd.read_csv("augmentation_B.csv")
df_C = pd.read_csv("augmentation_C.csv")

# extending the dataset with real A and B candidates
df = pd.concat([df, df_A, df_B, df_C], ignore_index=True)
df["label"] = df["label"].replace({"A": "AB", "B": "AB"})
df = df.dropna(subset=["label"])
print(df["label"].value_counts())

plt.figure(figsize=(5, 4))
df["label"] = pd.Categorical(df["label"], categories=["AB", "C", "D"], ordered=True)
sns.countplot(data=df, x="label")
plt.title("Label Distribution")
plt.savefig("label_distribution.png")

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


# smote = SMOTE(random_state=random_state)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# print("Before SMOTE:", y_train.value_counts().to_dict())
# print("After SMOTE:", y_train_balanced.value_counts().to_dict())

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

rus = RandomUnderSampler(random_state=random_state)
X_train_balanced, y_train_balanced = rus.fit_resample(X_train_scaled, y_train)
print("Before Undersampling:", Counter(y_train).items())
print("After Undersampling:", Counter(y_train_balanced).items())

nb = GaussianNB()
nb.fit(X_train_balanced, y_train_balanced)
y_pred_nb = nb.predict(X_test_scaled)
print("\n[Naive Bayes]")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train_balanced, y_train_balanced)
y_pred_svm = svm.predict(X_test_scaled)
print("\n[SVM]")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


scores_nb = cross_val_score(nb, X_scaled, y, cv=5)
scores_svm = cross_val_score(svm, X_scaled, y, cv=5)

print(f"\n[CV Accuracy] Naive Bayes: {scores_nb.mean():.3f} ± {scores_nb.std():.3f}")
print(f"[CV Accuracy] SVM:         {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")

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
# smote = SMOTE(random_state=random_state)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# # Undersampling
rus = RandomUnderSampler(random_state=random_state)
X_train_balanced, y_train_balanced = rus.fit_resample(X_train_scaled, y_train)


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
plt.savefig("top_10_features_rf.png")
# plt.show()

joblib.dump(rf, "best_classifier_rf.joblib")
joblib.dump(selected_feature_names.tolist(), "selected_features_rf.joblib")

X_test_full = X_test.reset_index(drop=True)
y_test_full = y_test.reset_index(drop=True)
y_pred_full = pd.Series(y_pred_rf, name="predicted")
df_plot = pd.concat([X_test_full, y_test_full, y_pred_full], axis=1)
df_plot["correct"] = df_plot["label"] == df_plot["predicted"]

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

cm = confusion_matrix(y_test, y_pred_voting, labels=["A", "B", "C", "D"])

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
plt.title("Confusion Matrix Heatmap for Voting Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_voting.png")

# center = average latitude and longitude
map_center = [(df["latitude"].max() + df["latitude"].min()) / 2, (df["longitude"].max() + df["longitude"].min()) / 2]
m = folium.Map(location=map_center, zoom_start=6)
heatmaps = {}
for c in df["label"].unique():
    df_d = df[df["label"] == c]
    heatmaps[c] = HeatMap(
        data=df_d[["latitude", "longitude"]].values, radius=15, blur=10, max_zoom=1
    )
    heatmaps[c].add_to(folium.FeatureGroup(name=c).add_to(m))

folium.LayerControl().add_to(m)
m.save("heatmaps_all.html")
