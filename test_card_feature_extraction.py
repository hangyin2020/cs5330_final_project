import cv2
import numpy as np
from fp_card_segmentation_functions import card_segmentation


# image = cv2.imread("./bulbasaur_normal.jpg")

# image = card_segmentation(image)
# # show the original image
# cv2.imshow("original", image)
# cv2.waitKey(0)




# show the grayscale image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("grayscale", gray)

# try clahe
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(gray)
# cv2.imshow("clahe", cl1)


# show the image with edges
# edges = cv2.Canny(gray, 90, 150, apertureSize=3)

def edge_filter(edges):
    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"number of contours is: {len(contours)}")

    # Define a minimum contour length threshold
    min_contour_length = 100  # Adjust this value based on your requirements

    # Filter out small contours
    filtered_edges = np.zeros_like(edges)
    for contour in contours:
        if cv2.arcLength(contour, closed=True) >= min_contour_length:
            cv2.drawContours(filtered_edges, [contour], -1, 255, thickness=2)

    return filtered_edges

def contours_extraction(image):
    image = card_segmentation(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    filtered_edges = edge_filter(edges)
    return filtered_edges

# filtered_edges = edge_filter(edges)
# Show the filtered edges
# cv2.imshow("Filtered Edges", filtered_edges)

# show the clahe image

# show the find contour image after clahe
# cv2.waitKey(0)
