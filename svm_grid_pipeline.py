import csv
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, Normalizer, MaxAbsScaler, minmax_scale
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import defaultdict
import warnings

import tqdm
warnings.filterwarnings("ignore")


scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'maxabs': MaxAbsScaler(),
}

kernels = ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed']

weight_strategies = {
    'balanced': 'balanced',
    'manual_0': {'A': 5, 'B': 3, 'C': 2, 'D': 1},
    'manual_1': {'A': 4, 'B': 3, 'C': 2, 'D': 1},
    'manual_fibonacci': {'A': 3, 'B': 2, 'C': 1, 'D': 1}
}

merge_strategies = {
    'none': None,
    'AB': {'A': 'AB', 'B': 'AB', 'C': 'C', 'D': 'D'},
    'ABC': {'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'D'},
    'ABnCD': {'A': 'AB', 'B': 'AB', 'C': 'nCD', 'D': 'nCD'},
}

sampling_strategies = {
    'none': None,
    'smote': SMOTE(random_state=42),
    'undersample': RandomUnderSampler(random_state=42)
}

# ==========

df = pd.read_csv("results_pt_all_combined.csv")
df = df[df["zoom"] == 12]

labels = {}
with open("l_pt_12.csv", "r") as f:
    reader = csv.reader(f)
    for rows in reader:
        labels[rows[0]] = rows[1]
df["label"] = df["city_name"].map(labels)
df = df.dropna(subset=["label"])

print(df.head())
print(df["label"].value_counts())

# adding data from other countries for A and B labels to balance the dataset
df_A = pd.read_csv("augmentation_A.csv")
df_B = pd.read_csv("augmentation_B.csv")
df_C = pd.read_csv("augmentation_C.csv")

merge_map = {'A': 'AB', 'B': 'AB', 'C': 'C', 'D': 'D'}
df = pd.concat([df, df_A, df_B, df_C], ignore_index=True)
df = df.dropna(subset=["label"])
df['label'] = df['label'].replace(merge_map)
print(df["label"].value_counts())

# ==========
df = df.groupby('label').apply(lambda x: x.sample(n=200)).reset_index(drop=True)
X_all = df.drop(columns=['label', 'city_name', 'latitude', 'longitude', 'zoom_lat', 'zoom_lon', 'zoom'])
y_original = df['label']

results = []

for merge_name, merge_map in merge_strategies.items():
    y = y_original.copy()
    if merge_map:
        y = y.replace(merge_map)

    valid_classes = y.unique().tolist()

    for sampler_name, sampler in sampling_strategies.items():
        X_train, X_test, y_train, y_test = train_test_split(X_all, y, stratify=y, test_size=0.2, random_state=42)

        for scaler_name, scaler in scalers.items():
            for weight_name, weight in weight_strategies.items():
                if isinstance(weight, dict):
                    weight = {cls: w for cls, w in weight.items() if cls in valid_classes}

                for kernel in kernels:
                    if kernel == 'precomputed':
                        continue

                    clf = SVC(kernel=kernel, class_weight=weight)
                    steps = []
                    if sampler_name != 'none':
                        steps.append(('sampler', sampler))
                    steps.append(('scaler', scaler))
                    steps.append(('svm', clf))

                    pipeline = ImbPipeline(steps)

                    param_grid = {
                        'svm__C': [0.1, 1, 10],
                        'svm__gamma': ['scale', 'auto']
                    }

                    try:
                        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
                        grid.fit(X_train, y_train)
                        y_pred = grid.predict(X_test)

                        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
                        macro_f1 = f1_score(y_test, y_pred, average='macro')

                        results.append({
                            'merge_strategy': merge_name,
                            'sampler': sampler_name,
                            'scaler': scaler_name,
                            'class_weight': weight_name,
                            'kernel': kernel,
                            'best_params': grid.best_params_,
                            'cv_score': grid.best_score_,
                            'test_weighted_f1': weighted_f1,
                            'test_macro_f1': macro_f1
                        })

                        print(f"[+] merge={merge_name}, sampler={sampler_name}, scaler={scaler_name}, weight={weight_name}, kernel={kernel} | F1: {weighted_f1:.3f}")
                    except Exception as e:
                        print(f"[-] merge={merge_name}, sampler={sampler_name}, scaler={scaler_name}, weight={weight_name}, kernel={kernel} | ERROR: {e}")

# ==== SAVE RESULTS ====
results_df = pd.DataFrame(results)
results_df.to_csv("svm_grid_results.csv", index=False)
print("Saved grid search results to svm_grid_results.csv")