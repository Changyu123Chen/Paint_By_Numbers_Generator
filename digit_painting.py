from sklearn.cluster import KMeans
import cv2
import numpy as np

# 1. K-means clustering
# Load the image to process:
img = cv2.imread("test.JPG")
if img is None:
    print("Cannot open the file!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# RESHAPE THE img to a 2D array of pixels
pxl_val = img.reshape((-1, 3))
pxl_val = np.float32(pxl_val)

# Define number of colors (clusters)
n_colors = 16  # Keep the number of colors to 16
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pxl_val)
centers = np.uint8(kmeans.cluster_centers_)
labels = kmeans.labels_

# Convert all pixels to the color of the centroids
segmented_img = centers[labels.flatten()]
segmented_img = segmented_img.reshape(img.shape)

cv2.imwrite("output_step1_kmeans.png", cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

# 2. Group adjacent pixels of the same color to form "facets"
gray_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
_, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. Merge small facets into larger neighboring ones
min_area = 1500  # Define the threshold area for small facets, increase to remove more small details

# Create a mask for small regions
mask = np.zeros_like(gray_img)

for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
        # Draw the small region on the mask
        cv2.drawContours(mask, [contour], -1, 255, -1)

# Dilate the mask to connect small regions with larger neighboring regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # Use a larger kernel for merging
mask = cv2.dilate(mask, kernel, iterations=2)

# Apply the mask to the segmented image to merge small facets with larger areas
merged_img = segmented_img.copy()
merged_img[mask == 255] = [0, 0, 0]  # Set small regions to black for merging

# Use flood fill to blend the small regions into neighboring large regions
h, w = mask.shape
flood_fill_img = merged_img.copy()
for y in range(h):
    for x in range(w):
        if mask[y, x] == 255:
            cv2.floodFill(flood_fill_img, mask, (x, y), (int(centers[labels[y * w + x]][0]),
                                                          int(centers[labels[y * w + x]][1]),
                                                          int(centers[labels[y * w + x]][2])))

# Save the merged facets result
cv2.imwrite("output_step3_merged_facets.png", cv2.cvtColor(flood_fill_img, cv2.COLOR_RGB2BGR))

# 4. Smooth the merged image to reduce sharp edges
smoothed_img = cv2.medianBlur(flood_fill_img, 7)  # Use median blur to smooth the facets
cv2.imwrite("output_step4_smoothed_facets.png", cv2.cvtColor(smoothed_img, cv2.COLOR_RGB2BGR))
