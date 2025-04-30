# That's a tool for converting admin.bounds to KML files.
import csv
import os
import simplekml
import pandas as pd

# Data - Copyright (C) 2025 by Curiosio
# Licensed under ECL (Educational Community License)
file_path = "ds_pt.tsv"

labels_file_path = "l_pt_12.csv"

BY_CLASS = True  # False - by area


def extract_coordinates(bbox):
    try:
        coords = bbox[9:-2].split(",")
        coords = [tuple(map(float, coord.split())) for coord in coords]
        return coords
    except Exception as e:
        print(f"Error processing bounding box: {bbox}, Error: {e}")
        return None


def process_tsv(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    dft = pd.read_csv(file_path, sep="\t", skiprows=3)
    expected_columns = ["osm_id", "name", "country", "area", "bounding_box"]
    for col in expected_columns:
        if col not in dft.columns:
            raise ValueError(f"Missing expected column: {col}")
    dft["bounding_box"] = dft["bounding_box"].apply(extract_coordinates)  # (lon, lat)
    return dft


try:
    df_ = process_tsv(file_path)

    if BY_CLASS:
        with open(labels_file_path, "r") as f:
            reader = csv.reader(f)
            labeled = {rows[0]: rows[1] for rows in reader}

        # by label
        # if id in labeled:
        df_A = df_[df_["osm_id"].isin([int(k) for k, v in labeled.items() if v == "A"])]
        df_B = df_[df_["osm_id"].isin([int(k) for k, v in labeled.items() if v == "B"])]
        df_C = df_[df_["osm_id"].isin([int(k) for k, v in labeled.items() if v == "C"])]
        df_D = df_[df_["osm_id"].isin([int(k) for k, v in labeled.items() if v == "D"])]

        dfs = {
            "A": (df_A, simplekml.Color.red),
            "B": (df_B, simplekml.Color.magenta),
            "C": (df_C, simplekml.Color.yellow),
            "D": (df_D, simplekml.Color.blue),
        }
    else:
        df_s = df_[df_["area"] < 1]
        df_m = df_[(df_["area"] >= 1) & (df_["area"] < 10)]
        df_l = df_[(df_["area"] >= 10) & (df_["area"] < 100)]
        df_xl = df_[(df_["area"] >= 100) & (df_["area"] < 1000)]
        df_xxl = df_[(df_["area"] >= 1000) & (df_["area"] < 10000)]
        df_xxxl = df_[df_["area"] >= 10000]
        dfs = {
            "s": (df_s, simplekml.Color.blue),
            "m": (df_m, simplekml.Color.yellow),
            "l": (df_l, simplekml.Color.red),
            "xl": (df_xl, simplekml.Color.magenta),
            "xxl": (df_xxl, simplekml.Color.orange),
            "xxxl": (df_xxxl, simplekml.Color.black),
        }

except Exception as e:
    print(f"Error processing file: {e}")

image_folders = [
    "results_pt_all",
]


def get_images(folder, cropped=True):
    images = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    print(len(images))
    if cropped:
        images_with_data = sorted(
            [
                f.split("_")[0] + "_network_cropped.png"
                for f in os.listdir(folder)
                if f.endswith(".csv")
            ]
        )
    else:
        images_with_data = sorted(
            [
                f.split("_")[0] + "_network.png"
                for f in os.listdir(folder)
                if f.endswith(".csv")
            ]
        )
    print(len(images_with_data))
    images = sorted(set(images) & set(images_with_data))
    print(len(images))
    return images


def get_images_from_folders(folders):
    images = {}
    for folder in folders:
        ims = get_images(folder)
        images[folder] = ims
    image_path_by_osm_id = {}
    for folder, ims in images.items():
        for img in ims:
            osm_id = int(img.split("_")[0])
            image_path_by_osm_id[osm_id] = os.path.join(folder, img)
            # making sure the image path is absolute
            if not os.path.isabs(image_path_by_osm_id[osm_id]):
                image_path_by_osm_id[osm_id] = os.path.abspath(
                    image_path_by_osm_id[osm_id]
                )
    return image_path_by_osm_id


def create_kml(dfs, output_file, images):
    kml = simplekml.Kml()
    for key, (df, color) in dfs.items():
        # creating a folder for each key
        folder = kml.newfolder(name=key)
        for index, row in df.iterrows():
            coords = row["bounding_box"]
            if coords is not None:
                polygon = folder.newpolygon(
                    name=f"{row['osm_id']}_{row['name']}_{row['country']}_{row['area']}",
                    outerboundaryis=coords,
                )
                polygon.style.linestyle.color = color
                polygon.style.linestyle.width = 2
                polygon.style.polystyle.color = simplekml.Color.changealphaint(
                    100, color
                )
                polygon.style.polystyle.fill = 0

                # adding a url to the polygon
                polygon.description = f"""
                <a href="https://www.openstreetmap.org/relation/{row['osm_id']}">OpenStreetMap</a><br>
                """

                # search for the image in images
                if row["osm_id"] in images:
                    image = images[row["osm_id"]]
                    if os.path.exists(image):
                        polygon.description += f"""
                        <img src="{image}" width="400" height="300">
                        """
                        go = folder.newgroundoverlay(name=row["osm_id"])
                        go.icon.href = image
                        lons = [point[0] for point in coords]
                        lats = [point[1] for point in coords]
                        go.latlonbox.north = max(lats)
                        go.latlonbox.south = min(lats)
                        go.latlonbox.east = max(lons)
                        go.latlonbox.west = min(lons)
                        go.style.liststyle.listitemtype = simplekml.DisplayMode.hide
                    else:
                        print(f"Image not found: {image}")
    kml.save(output_file)


if __name__ == "__main__":
    images = get_images_from_folders(image_folders)
    create_kml(dfs, "pt_labels.kml", images)
