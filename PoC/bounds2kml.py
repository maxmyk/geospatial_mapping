# That's a tool for converting admin.bounds to KML files.
import csv
import os
import simplekml
import pandas as pd


def extract_coordinates(bbox):
    try:
        coords = bbox[9:-2].split(",")
        coords = [tuple(map(float, coord.split())) for coord in coords]
        return coords
    except Exception as e:
        print(f"Error processing bounding box: {bbox}, Error: {e}")
        return None


def process_tsv(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    # Read the TSV file into a DataFrame
    dft = pd.read_csv(file_path, sep="\t", skiprows=3)
    # osm_id	name	country	area	bounding_box
    expected_columns = ["osm_id", "name", "country", "area", "bounding_box"]
    for col in expected_columns:
        if col not in dft.columns:
            raise ValueError(f"Missing expected column: {col}")

    dft["bounding_box"] = dft["bounding_box"].apply(extract_coordinates)  # (lon, lat)

    return dft

# Data - Copyright (C) 2025 by Curiosio
# Licensed under ECL (Educational Community License)
# file_path = "ds_12k.tsv"
file_path = "PoC/ds.tsv"

try:
    df_ = process_tsv(file_path)
    # df_l = df_[8000:12000]
    df_l = df_[1000:1500]
    # df_m = df_[4000:8000]
    df_m = df_[500:1000]
    # df_s = df_[0:4000]
    df_s = df_[0:500]

    dfs = {
        "s": (df_s, simplekml.Color.violet),
        "m": (df_m, simplekml.Color.yellow),
        "l": (df_l, simplekml.Color.red),
    }

except Exception as e:
    print(f"Error processing file: {e}")

# going through dfs and creating kml file with multiple polygons

image_folders = [
    "PoC/results_m_drive_alt",
    "PoC/results_s_all_alt",
]

def get_images(folder):
    images = sorted([f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")])
    print(len(images))
    images_with_data = sorted([f.split("_")[0]+"_network_all.png" for f in os.listdir(folder) if f.endswith(".csv")])
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
                image_path_by_osm_id[osm_id] = os.path.abspath(image_path_by_osm_id[osm_id])
    return image_path_by_osm_id

def create_kml(dfs, output_file, images):
    kml = simplekml.Kml()
    for key, (df, color) in dfs.items():
        # creating a folder for each key
        folder = kml.newfolder(name=key)
        for index, row in df.iterrows():
            coords = row["bounding_box"]
            if coords is not None:
                polygon = folder.newpolygon(name=f"{row['osm_id']}_{row['name']}_{row['country']}_{row['area']}",
                                             outerboundaryis=coords)
                polygon.style.linestyle.color = color
                polygon.style.linestyle.width = 2
                polygon.style.polystyle.color = simplekml.Color.changealphaint(100, color)
                polygon.style.polystyle.fill = 0

                # adding a url to the polygon
                polygon.description = f"""
                <a href="https://www.openstreetmap.org/relation/{row['osm_id']}">OpenStreetMap</a><br>
                """

                # adding a screenshot to the polygon
                # search for the image in images
                if row["osm_id"] in images:
                    image = images[row["osm_id"]]
                    if os.path.exists(image):
                        polygon.description += f"""
                        <img src="{image}" width="400" height="300">
                        """
                        go = folder.newgroundoverlay(name=row["osm_id"])
                        go.icon.href = image
                        go.latlonbox.north = coords[0][1]
                        go.latlonbox.south = coords[2][1]
                        go.latlonbox.east = coords[1][0]
                        go.latlonbox.west = coords[0][0]

                    else:
                        print(f"Image not found: {image}")

    kml.save(output_file)

images = get_images_from_folders(image_folders)
create_kml(dfs, "ds.kml", images)