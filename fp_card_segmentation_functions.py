import cv2
import numpy as np

def point_to_line_distance(x0, y0, x1, y1, x2, y2):
    """
    Calculate the distance between point (x0, y0) and line segment (x1, y1)-(x2, y2).
    This function is used to detect if two lines have same slope and also the distance
    of the two lines are close enough to be used to identify one edge on the image.
    """
    # Numerator of the distance formula
    # The area of the triangle
    numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    # the length of the line
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Calculate distance
    distance = numerator / denominator
    return distance

def line_intersection(line1, line2):
    """
    Function to calculate the intersection of two lines.
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    
    # Calculate slopes
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')
    
    # Check for parallel lines
    if m1 == m2:
        return float('inf'), float('inf')  # Lines are parallel, no intersection
    
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

def card_segmentation(image):
    """
    Function to find the region of a card in a given image
    input: an image including background and a card
    output: a rectangle that only include the card

    Hough Transform is used to find 4 lines that represents the sides 
    of the card. This is done by finding the largest contour first
    and then use Hough Transform to find the lines of the contour. The 
    lines are added into a set of lines if they have different slopes or
    two lines have same slopes but different position, for example, top side
    and bottom side.
    Then we calculate the intersection of those 4 lines to get 4 points as
    the corners of the warped image, and finally we project the card region
    to the warped image.
    """
    # Read the image
    # for debug only: image = cv2.imread("./data/pokemon_cards/caz_oddish.png")
    if image is None:
        print("can't read the image")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to filter out the noise
    blur_img = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold or perform edge detection to obtain a binary image
    # Example using Canny edge detection
    edges = cv2.Canny(blur_img, 50, 150, apertureSize=3)

    # Dilate the edges to make them thicker
    kernel_size = 5  # Choose a larger kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)


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

    # return if contour cannot be found
    if largestContour is None:
        print("cannot find a contour")
        return None
    
    # calculate the largestArea
    largestArea = cv2.contourArea(largestContour)

    # If the area of the contour is less than 1000, the contour is too small
    if largestArea < 1000:
        print("The contour is too small")
        return None
    
    # Draw the largestContour on the image
    # cv2.drawContours(image, [largestContour], -1, (0, 255, 0), 2)

    # Create a blank canvas to draw the largest contour
    canvas = np.zeros_like(gray)
    cv2.drawContours(canvas, [largestContour], -1, (255), thickness=4)

    # Detect lines using Hough Line Transform of the largest contour
    lines = cv2.HoughLinesP(canvas, 1, np.pi / 180, threshold=200, minLineLength=300, maxLineGap=10)

    # return if no lines are detected
    if lines is None:
        print("lines are not detected")
        return None

    # Initialize slopes and sides. 
    # slopes include all the 4 possible sides of a rectangle, 2 vertical and 2 horizontal
    # the sides is a list of 4 lines. Each line is represented by x1, y1, x2, y2.
    slopes = []
    sides = []

    # Select 4 lines that represent the card sides
    if lines is not None and len(lines) >= 4:
        # sort the lines by length
        lines = sorted(lines, key=lambda x: np.linalg.norm(x[0][0:2] - x[0][2:]), reverse=True)

        # Add lines into list, if that line has a diff slope with the existing slopes, 
        # or big diff position of distance even if slope is same (two opposite sides)
        for line in lines:
            # flags for slope found and side found
            found_slope = False
            found_side = False
            # grab the points from line
            x1, y1, x2, y2 = line[0]
            # if x are too close, assume the slope is infinity
            if abs(x2-x1) < 10:
                curr_slope = 10000
            # if y are too close, assume the slope is 0
            elif abs(y2 - y1) < 10:
                curr_slope = 0
            # else calculate the slope
            else: 
                curr_slope = (y2-y1) / (x2-x1)
            # if it is the first slope, we add it to the slopes list
            if len(slopes) == 0:
                slopes.append(curr_slope)
            # Check if we already have a similar slope in the list
            for slope in slopes:
                if abs(curr_slope - slope) < 10:
                    found_slope = True
            # Check if we find a new side
            for side in sides:
                # grab the points of the side
                px1, py1, px2, py2 = side[0]
                # calculate the distance between the point on the existing side
                # with the current line
                distance = point_to_line_distance(px1, py1, x1, y1, x2, y2)
                # if slope is similar and the distance is close,
                # it means they are treated to be on the same line
                # and set found side to be true
                if found_slope and distance < 10:
                    found_side = True
            # if the side is a new line, we add it to the side list
            if (found_side is False):
                sides.append(line)
            # if the slope is a slope, we add it to the slope list
            if (found_slope is False):
                slopes.append(curr_slope)

    # If sides is none or not enough sides found, return none
    if sides is None or len(sides) != 4:
        print("sides can't be detected or not enough sides detected")
        return None
    
    # Draw the detected lines on the original image
    for side in sides:
        x1, y1, x2, y2 = side[0]
        # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    # Initialize 4 sides of the card
    left_side = []
    right_side = []
    top_side = []
    bottom_side = []

    # How to tell which side is which?
    # sort the sides with average x, the min average x is left side,
    # and the max average x is right side
    sides = sorted(sides, key=lambda x: x[0][0] + x[0][2])
    left_side = sides[0]
    right_side = sides[3]

    # sort the sides with average y, the min average y is top side,
    # and the max average y is bottom side
    sides = sorted(sides, key=lambda x: x[0][1] + x[0][3])
    top_side = sides[0]
    bottom_side = sides[3]

    # Find the intersections of the 4 sides
    # Here we start from the top left corner and the direction is counter-clockwise
    # This direction has to be consistent with the dst_points below
    intersection0_x, intersection0_y = line_intersection(left_side, top_side)
    intersection1_x, intersection1_y = line_intersection(left_side, bottom_side)
    intersection2_x, intersection2_y = line_intersection(right_side, bottom_side)
    intersection3_x, intersection3_y = line_intersection(right_side, top_side)

    # return none if any intersection is not found
    if (intersection0_x == float('inf') or 
        intersection1_x == float('inf') or
        intersection2_x == float('inf') or
        intersection3_x == float('inf') or
        intersection0_y == float('inf') or
        intersection1_y == float('inf') or
        intersection2_y == float('inf') or
        intersection3_y == float('inf')):
        print("intersections are not detected")
        return None

    # get the int value for the pixel
    intersection0 = (int(intersection0_x), int(intersection0_y))
    intersection1 = (int(intersection1_x), int(intersection1_y))
    intersection2 = (int(intersection2_x), int(intersection2_y))
    intersection3 = (int(intersection3_x), int(intersection3_y))

    # Draw the point on the image
    # cv2.circle(image, intersection0, 5, (255), -1)  # -1 means filled circle, use positive values for outline only
    # cv2.circle(image, intersection1, 5, (255), -1)  # -1 means filled circle, use positive values for outline only
    # cv2.circle(image, intersection2, 5, (255), -1)  # -1 means filled circle, use positive values for outline only
    # cv2.circle(image, intersection3, 5, (255), -1)  # -1 means filled circle, use positive values for outline only

    # The points list
    points = []
    points.append(intersection0)
    points.append(intersection1)
    points.append(intersection2)
    points.append(intersection3)
    
    # Convert the list to a NumPy array
    points_array = np.array(points, dtype=np.float32)
    
    # Define the destination points for the perspective transform
    width = 786  # Define the width of the output rectangle
    height = 1110  # Define the height of the output rectangle
    # counter-clockwise assign to dst
    dst_points = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    # Perform perspective transform
    matrix = cv2.getPerspectiveTransform(points_array.astype(np.float32), dst_points)
    warped_image = cv2.warpPerspective(image, matrix, (width, height))

    return warped_image
