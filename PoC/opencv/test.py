import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('Hamburg_network.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('Hamburg_network1.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
print(f"Keypoints in img1: {len(kp1)}")

kp2, des2 = orb.detectAndCompute(img2, None)
print(f"Keypoints in img2: {len(kp2)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
raw_matches = bf.knnMatch(des1, des2, k=2)

ratio_thresh = 0.85
good_matches = []
for m, n in raw_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

print(f"Matches after Lowe's ratio test: {len(good_matches)}")

if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel().tolist()
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    outlier_matches = [m for i, m in enumerate(good_matches) if not mask[i]]
else:
    inlier_matches = []
    outlier_matches = []
    print("Not enough matches for homography estimation.")

print(f"Inliers after RANSAC: {len(inlier_matches)}")
print(f"Outliers removed: {len(outlier_matches)}")

h1, w1 = img1.shape
h2, w2 = img2.shape
out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
out_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
out_img[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

def draw_matches(matches, color):
    for m in matches:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
        cv2.line(out_img, pt1, pt2_shifted, color, 1)

draw_matches(inlier_matches, (0, 255, 0))
draw_matches(outlier_matches, (0, 0, 255))

plt.figure(figsize=(14, 7))
plt.imshow(out_img[..., ::-1])
plt.title("Matches after Lowe's Ratio Test + RANSAC: Green = Inliers, Red = Outliers")
plt.axis('off')
plt.show()
