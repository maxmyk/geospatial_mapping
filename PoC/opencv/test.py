import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import gridspec
from tqdm import tqdm


def compute_matches(detector, img1, img2, ratio_thresh=0.85, ransac_thresh=5.0):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return [], [], kp1, kp2

    if (
        isinstance(detector, cv2.ORB)
        or isinstance(detector, cv2.AKAZE)
        or isinstance(detector, cv2.BRISK)
    ):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2)

    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return [], [], kp1, kp2

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if mask is None:
        return [], [], kp1, kp2

    mask = mask.ravel().tolist()
    inliers = [m for i, m in enumerate(good_matches) if mask[i]]
    outliers = [m for i, m in enumerate(good_matches) if not mask[i]]
    return inliers, outliers, kp1, kp2


def draw(match_list, color, kp1, kp2, out_img, w1):
    for m in match_list:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
        cv2.line(out_img, pt1, pt2_shifted, color, 1)
    return out_img


def draw_matches(img1, img2, kp1, kp2, inliers, outliers):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    out_img[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    out_img = draw(inliers, (0, 255, 0), kp1, kp2, out_img, w1)
    out_img = draw(outliers, (0, 0, 255), kp1, kp2, out_img, w1)
    return out_img


def resize_image(img, width=1600, resize_float=False):
    if resize_float:
        img = img.astype(np.float32) / 255.0
    h, w = img.shape
    scale = width / w
    resized = cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) if resize_float else resized


def main(img1_path, img2_path, iter, results_list):
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        raise FileNotFoundError("Images not found")

    img1 = cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = resize_image(img1, width=1600, resize_float=True)
    img2 = resize_image(img2, width=1600, resize_float=True)

    # converting to uint8 Grayscale
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    results = {}
    ratios = {}

    for name, detector in descriptors.items():
        inliers, outliers, kp1, kp2 = compute_matches(detector, img1, img2)
        matched_img = draw_matches(img1, img2, kp1, kp2, inliers, outliers)
        results[name] = (matched_img, (inliers, outliers, kp1, kp2))
        total = len(inliers) + len(outliers)
        ratios[name] = len(inliers) / total if total > 0 else 0

        results_list.append(
            {
                "image_pair": f"{os.path.basename(img1_path)} vs {os.path.basename(img2_path)}",
                "descriptor": name,
                "kp1": len(kp1),
                "kp2": len(kp2),
                "inliers": len(inliers),
                "outliers": len(outliers),
                "inlier_ratio": ratios[name],
            }
        )

    fig1 = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2)
    for i, (name, data) in enumerate(results.items()):
        img, (il, ol, k1, k2) = data
        ax = fig1.add_subplot(gs[i])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{name}: {len(il)} inliers, {len(ol)} outliers, {len(k1)} & {len(k2)} keypoints"
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(img1_path.replace(".png", f"_desc_{iter}.png"))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(ratios.keys(), [r * 100 for r in ratios.values()])
    ax2.set_title("Inlier Ratio by Descriptor")
    ax2.set_ylabel("Inlier Ratio (%)")
    # ax2.set_ylim(0, 100)
    ax2.set_yscale("log")
    ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    ax2.grid(True)
    plt.savefig(img1_path.replace(".png", f"_desc_{iter}_ratios.png"))


if __name__ == "__main__":
    results_list = []

    images = [
        ("PoC/A/13251349_A_bw_osm.png", "PoC/A/13251349_A_bw_sat.png"),
        # ("PoC/A/13251349_A_bw_osm.png", "PoC/A/13251349_A_bw_sat_roads.png"),
        ("PoC/A/5989554_A_bw_osm.png", "PoC/A/5989554_A_bw_sat.png"),
        # ("PoC/A/5989554_A_bw_osm.png", "PoC/A/5989554_A_bw_sat_roads.png"),
        ("PoC/A/6384111_A_bw_osm.png", "PoC/A/6384111_A_bw_sat.png"),
        # ("PoC/A/6384111_A_bw_osm.png", "PoC/A/6384111_A_bw_sat_roads.png"),
        ("PoC/B/6427489_B_bw_osm.png", "PoC/B/6427489_B_bw_sat.png"),
        # ("PoC/B/6427489_B_bw_osm.png", "PoC/B/6427489_B_bw_sat_roads.png"),
        ("PoC/B/1391324_B_bw_osm.png", "PoC/B/1391324_B_bw_sat.png"),
        # ("PoC/B/1391324_B_bw_osm.png", "PoC/B/1391324_B_bw_sat_roads.png"),
        ("PoC/C/6386158_C_bw_osm.png", "PoC/C/6386158_C_bw_sat.png"),
        # ("PoC/C/6386158_C_bw_osm.png", "PoC/C/6386158_C_bw_sat_roads.png"),
        ("PoC/C/1391333_C_bw_osm.png", "PoC/C/1391333_C_bw_sat.png"),
        # ("PoC/C/1391333_C_bw_osm.png", "PoC/C/1391333_C_bw_sat_roads.png"),
        ("PoC/D/4112279_D_bw_osm.png", "PoC/D/4112279_D_bw_sat.png"),
        # ("PoC/D/4112279_D_bw_osm.png", "PoC/D/4112279_D_bw_sat_roads.png"),
        ("PoC/D/12905750_D_bw_osm.png", "PoC/D/12905750_D_bw_sat.png"),
        # ("PoC/D/12905750_D_bw_osm.png", "PoC/D/12905750_D_bw_sat_roads.png"),
    ]

    descriptors = {
        "SIFT": cv2.SIFT_create(),
        "AKAZE": cv2.AKAZE_create(),
        "BRISK": cv2.BRISK_create(),
        "KAZE": cv2.KAZE_create(),
    }

    for i, (image1, image2) in tqdm(enumerate(images)):
        main(image1, image2, i, results_list)

    with open("features_detected.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "image_pair",
                "descriptor",
                "kp1",
                "kp2",
                "inliers",
                "outliers",
                "inlier_ratio",
            ],
        )
        writer.writeheader()
        for row in results_list:
            writer.writerow(row)
