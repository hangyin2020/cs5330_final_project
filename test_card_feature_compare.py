import cv2
import numpy as np
from test_card_feature_extraction import contours_extraction
from fp_card_segmentation_functions import card_segmentation


image1 = cv2.imread("./bulbasaur_normal.jpg")
image2 = cv2.imread("./bulbasaur_tcg.jpg")
image1_segmentation = card_segmentation(image1)
image2_segmentation = card_segmentation(image2)
cv2.imshow("image1_segmentation", image1_segmentation)
cv2.imshow("image2_segmentation", image2_segmentation)


image1_contours = contours_extraction(image1)
image2_contours = contours_extraction(image2)
cv2.imshow("image1_edges", image1_contours)
cv2.imshow("image2_edges", image2_contours)

# Get the dimensions of image1_contours
height, width = image1_contours.shape[:2]



# Resize image2_contours to match the size of image1_contours
# image2_contours_resized = cv2.resize(image2_contours, (width1, height1))

# Create an empty canvas of the same size as image1
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Draw edges from the first image (in green)
canvas[image1_contours > 0] = (0, 255, 0)

# Draw edges from the second image (in blue)
canvas[image2_contours > 0] = (255, 0, 0)

# Display the canvas with both contours
cv2.imshow("Contours", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

