# Geospatial Mapping of Settlement Patterns for Aerovisual Navigation
This is the codebase of my thesis. It investigates how geospatial characteristics of settlements can be used for effcient aerovisual navigation.

## How-to-Use Steps:

### 0. Install Dependencies
Run ```pip install -r requirements.txt```

### 1. Features Extraction
Run ```python combined.py```. Can take up to *8 GB* of storage!

### 2. Labeling
Run ```python label_images.py```. SNP (Street Network Plot) and corresponding OSM_ID show up. Pressing A, B, C, or D assigns a corresponding label.

You can run several labeling rounds and then get median values using ```utils/labeling_merger.py```.

### 3. Combining Data With Labels
Run ```python utils/combiner.py```. Combines features with labels.

### 4. Visualizing The Data for Google Earth
Run ```python utils/images_cropper.py``` & ```python utils/bounds2kml.py```. Crops SNPs correctly and creates a KML file for Google Earth. Running it with ```BY_CLASS = False``` makes labeling the settlements much easier.

### 5. SVM Training and Inference
Run ```python classification_new.py```. Trains the SVM with correct parameters, creates heatmaps.

## Misc.
- For CV descriptors test - see ```PoC/opencv/```.
- For SuperGlue test we used an official unmodified pretrained implementation - see [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).
- All previous tests with classifiers are located in ```PoC/classification.py```
- Heatmaps are generated at Inference step.

## Important!
- ```ds_pt.tsv``` - Copyright (C) 2025 by [Curiosio](https://curiosio.com/), Licensed under ECL (Educational Community License). Portuguese settlements dataset.
- Files used for dataset augmentation were derived using the same pipeline, but the initial ```*.tsv``` settlements dataset remains private.