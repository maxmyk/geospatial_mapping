import pandas as pd

df = pd.read_csv("svm_grid_results_combined.csv")

# removing "ABnCD" merge strategy
df = df[df["merge_strategy"] != "ABnCD"]

top_weighted = df.sort_values("test_weighted_f1", ascending=False).head(10)
top_macro = df.sort_values("test_macro_f1", ascending=False).head(10)

top_sampler = df.groupby("sampler")["test_weighted_f1"].mean().sort_values(ascending=False)
top_kernel = df.groupby("kernel")["test_weighted_f1"].mean().sort_values(ascending=False)
top_weight = df.groupby("class_weight")["test_weighted_f1"].mean().sort_values(ascending=False)
top_scaler = df.groupby("scaler")["test_weighted_f1"].mean().sort_values(ascending=False)
top_merge = df.groupby("merge_strategy")["test_weighted_f1"].mean().sort_values(ascending=False)

print("Top 10 Weighted F1 Scores:")
print(top_weighted[["merge_strategy", "sampler", "scaler", "class_weight", "kernel", "test_weighted_f1"]])
print("\nTop 10 Macro F1 Scores:")
print(top_macro[["merge_strategy", "sampler", "scaler", "class_weight", "kernel", "test_macro_f1"]])
print("\nTop Samplers:")
print(top_sampler)
print("\nTop Kernels:")
print(top_kernel)
print("\nTop Class Weights:")
print(top_weight)
print("\nTop Scalers:")
print(top_scaler)
print("\nTop Merge Strategies:")
print(top_merge)

