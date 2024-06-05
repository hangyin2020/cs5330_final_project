import cv2
import numpy as np

# Read the image
image = cv2.imread("./data/pokemon_cards/swablu.png")
cv2.imshow("original", image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("grayscale", gray)
cv2.waitKey(0)

blur_img = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold or perform edge detection to obtain a binary image
# Example using Canny edge detection
edges = cv2.Canny(blur_img, 50, 150, apertureSize=3)

# Dilate the edges to make them thicker
kernel_size = 5  # Choose a larger kernel size
kernel = np.ones((kernel_size, kernel_size), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

cv2.imshow("edges", dilated_edges)
cv2.waitKey(0)

# Find contours in the binary image
contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
maxArea = 0
largestContour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > maxArea:
        maxArea = area
        largestContour = contour

def point_to_line_distance(x0, y0, x1, y1, x2, y2):
    """
    Calculate the distance between point (x0, y0) and line segment (x1, y1)-(x2, y2).
    """
    # Numerator of the distance formula
    numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    # Denominator of the distance formula
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Calculate distance
    distance = numerator / denominator
    return distance

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    
    # Calculate slopes
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')
    
    # Check for parallel lines
    if m1 == m2:
        return None  # Lines are parallel, no intersection
    
    # Calculate intersection point
    if m1 == float('inf'):  # If line 1 is vertical
        x_intersect = x1
        y_intersect = m2 * (x1 - x3) + y3
    elif m2 == float('inf'):  # If line 2 is vertical
        x_intersect = x3
        y_intersect = m1 * (x3 - x1) + y1
    else:
        x_intersect = ((m1 * x1 - y1) - (m2 * x3 - y3)) / (m1 - m2)
        y_intersect = m1 * (x_intersect - x1) + y1
    
    return x_intersect, y_intersect


# Approximate the contour to a convex hull
if largestContour is not None:
    # cv2.drawContours(image, [largestContour], -1, (0, 255, 0), 2)

    # Show the modified image
    cv2.imshow("Largest Contour", image)
    cv2.waitKey(0)

    
    # Get the four corner points of the convex hull
    epsilon = 0.02 * cv2.arcLength(largestContour, True)
    approx = cv2.approxPolyDP(largestContour, epsilon, True)
    # Draw the polygon on the image
    # cv2.drawContours(image, [approx], 0, (255, 0, 255), 2)

    # Display the result
    # cv2.imshow("Polygon Approximation of Largest Contour", image)
    # cv2.waitKey(0)

    # Create a blank canvas to draw the largest contour
    canvas = np.zeros_like(gray)
    cv2.drawContours(canvas, [largestContour], -1, (255), thickness=4)

    # cv2.imshow("Polygon Approximation of Largest Contour", canvas)
    # cv2.waitKey(0)

    # Detect lines using Hough Line Transform within the largest contour
    lines = cv2.HoughLinesP(canvas, 1, np.pi / 180, threshold=200, minLineLength=300, maxLineGap=10)

    print(f"len of lines is: {len(lines)}")
    print(f"lines[0] is {lines[0]}")
    slopes = []
    sides = []
    # Sort and select only the longest 4 lines
    if lines is not None and len(lines) >= 4:
        lines = sorted(lines, key=lambda x: np.linalg.norm(x[0][0:2] - x[0][2:]), reverse=True)
        # here we can use something like, add a line into a list, if that line has a diff slope, 
        # or big diff position of distance even if slope is same (two opposite sides)
        for line in lines:
            found_slope = False
            found_side = False
            x1, y1, x2, y2 = line[0]
            if abs(x2-x1) < 10:
                curr_slope = 10000
            elif abs(y2 - y1) < 10:
                curr_slope = 0
            else: 
                curr_slope = (y2-y1) / (x2-x1)
            if len(slopes) == 0:
                slopes.append(curr_slope)
            for slope in slopes:
                if abs(curr_slope - slope) < 10:
                    found_slope = True
            for side in sides:
                px1, py1, px2, py2 = side[0]
                distance = point_to_line_distance(px1, py1, x1, y1, x2, y2)
                if found_slope and distance < 10:
                    found_side = True
            if (found_side is False):
                slopes.append(curr_slope)
                sides.append(line)

    print(f"len of sides {len(sides)}")
    # Draw the detected lines on the original image
    if sides is not None:
        for side in sides:
            x1, y1, x2, y2 = side[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    
    # sort the sides of average x, the min average x is left side,
    # and the max average x is right side
    sides = sorted(sides, key=lambda x: x[0][0] + x[0][2])

    left_side = []
    right_side = []
    top_side = []
    bottom_side = []
    
    left_side = sides[0]
    right_side = sides[3]
    sides = sorted(sides, key=lambda x: x[0][1] + x[0][3])
    top_side = sides[0]
    bottom_side = sides[3]


    intersection0_x, intersection0_y = line_intersection(left_side, top_side)
    intersection1_x, intersection1_y = line_intersection(left_side, bottom_side)
    intersection2_x, intersection2_y = line_intersection(right_side, bottom_side)
    intersection3_x, intersection3_y = line_intersection(right_side, top_side)
    print(f"x = {intersection0_x} and y = {intersection0_y}")
    intersection0 = (int(intersection0_x), int(intersection0_y))
    intersection1 = (int(intersection1_x), int(intersection1_y))
    intersection2 = (int(intersection2_x), int(intersection2_y))
    intersection3 = (int(intersection3_x), int(intersection3_y))
    # Draw the point on the image
    cv2.circle(image, intersection0, 5, (255), -1)  # -1 means filled circle, use positive values for outline only
    cv2.circle(image, intersection1, 5, (255), -1)  # -1 means filled circle, use positive values for outline only
    cv2.circle(image, intersection2, 5, (255), -1)  # -1 means filled circle, use positive values for outline only
    cv2.circle(image, intersection3, 5, (255), -1)  # -1 means filled circle, use positive values for outline only

    # Display the result
    cv2.imshow("Detected Lines within Largest Contour", image)
    cv2.waitKey(0)

    
    points = []
    points.append(intersection0)
    points.append(intersection1)
    points.append(intersection2)
    points.append(intersection3)
    
    # Convert the list to a NumPy array
    points_array = np.array(points, dtype=np.float32)
    # if len(approx) == 4:
    #     points = approx.reshape(4, 2)
    #     print(f"approx is {approx}")
    #     print(f"len of largestContour is {len(largestContour)}")
    #     print(f"points is {points}")
    # else:
    #     print("Could not detect four corners.")

    # Define the destination points for the perspective transform
    width = 786  # Define the width of the output rectangle
    height = 1110  # Define the height of the output rectangle
    dst_points = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    # Perform perspective transform
    matrix = cv2.getPerspectiveTransform(points_array.astype(np.float32), dst_points)
    warped_image = cv2.warpPerspective(image, matrix, (width, height))

    # Display the result
    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)
    cv2.imwrite("test_pokemon.png", warped_image)



cv2.destroyAllWindows()
