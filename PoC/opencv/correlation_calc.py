import pandas as pd

csv_file_path = "PoC/opencv/descs.csv"
with open(csv_file_path, "r") as f:
    csv_data = f.read()

from io import StringIO
df = pd.read_csv(StringIO(csv_data))

correlations = {
    descriptor.split()[0]: df[f"{descriptor} OSM"].corr(df[f"{descriptor} SAT"])
    for descriptor in ["SIFT", "AKAZE", "BRISK", "KAZE", "SuperPoint"]
}

correlation_df = pd.DataFrame(list(correlations.items()), columns=["Descriptor", "Pearson Correlation (OSM vs SAT)"])
correlation_df["Pearson Correlation (OSM vs SAT)"] = correlation_df["Pearson Correlation (OSM vs SAT)"].round(2)
print(correlation_df)
