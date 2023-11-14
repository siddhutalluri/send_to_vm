"""
This script contains some utility functions for the vision backend
"""
# importing the required libraries
import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import pandas as pd


def transform_bbox_to_original(cropped_bbox, crop_start_x, crop_start_y):
    # Unpack the cropped bounding box coordinates
    cropped_x1, cropped_y1, cropped_x2, cropped_y2 = cropped_bbox

    # Calculate the coordinates of the bounding box in the original image
    original_x1 = cropped_x1 + crop_start_x
    original_y1 = cropped_y1 + crop_start_y
    original_x2 = cropped_x2 + crop_start_x
    original_y2 = cropped_y2 + crop_start_y

    # Return the transformed bounding box coordinates
    return original_x1, original_y1, original_x2, original_y2


def drawAxis(img, p_, q_, colour,
             scale):  # this funtion will be used to draw axis/lines given its two set of points in an image with customised colur and thickness
    """ This functions 4 parameters
  1: img: image file
  2: p_ : 1st set of points where you want to draw axis
  3: q_ : 2nd set of points where you want to draw axis
  4: colour: a tuple of color in which you want to draw axis
  5: scale : thickness of the line you want to draw
  """
    img = img
    p_ = p_
    q_ = q_
    colour = colour
    scale = scale

    p = list(p_)
    q = list(q_)
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    ## [visualization1]


def getOrientation(pts, img):  # this function gets the orientation of the contours that is selected
    '''
  This function takes 2 parameters
  1: pts: points in which we need to find orientation
  2: img: image file
  '''

    cv2.imwrite("orientation.jpeg", img)
    pts = pts
    img = img
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    # cntr = (int(mean[0,0]), int(mean[0,1]))
    m = cv2.moments(pts)
    if m['m00'] != 0:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        cntr = (cx, cy)
    else:
        cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[1, 1], eigenvectors[1, 0])  # orientation in radians
    ## [visualization]
    angle = 180 - np.rad2deg(angle)
    cv2.imwrite("final_angle.jpeg", img)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)

    return angle, cntr


def measure_angle(src):
    # this function will return the angle by accepting the image path
    '''This function takes one parameter
      1: image file
      '''
    cv2.imwrite("initial_angle.jpeg", src)

    # src = cv.imread(path)
    if src is None:
        print('Could not open or find the image: ', src)
        exit(0)
    ## [pre-process]
    # Convert image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    ## [contours]
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    l = []  # list to store each contour

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        l.append([i, area, c])
        # Ignore contours that are too small or too large (you can this piece of code when you have mutiple objects in the image)
        # if area < 10000 or   1763769 < area:
        # continue
    df = pd.DataFrame(l, columns=['index', 'Area', "contour"])
    df = df.sort_values("Area", ascending=False)
    df.reset_index(inplace=True)
    n = df["index"][0]
    # Draw each contour only for visualisation purposes
    cv2.drawContours(src, contours, n, (0, 0, 255), 2)
    # Find the orientation of with largest shape
    c = df["contour"][0]
    # one can get the orientation of each contour if multiple objects are there.
    angle, center = getOrientation(c, src)
    # round the predicted angle upto 2 decimals
    angle = round(float(angle), 2)
    return angle, center


def non_max_suppression_fast(boxes, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def countour_detect_screws(image):
    original_image = image.copy()

    # cv2.imshow("Raw Image", image)
    # cv2.waitKey(0)

    # cropping_rect = [339, 19, 1171, 678]
    cropping_rect = [32, 18, 678, 399]

    # Rotated
    # cropping_rect = [203, 25, 611, 466]


    # Crop the image to the region of interest
    image = image[cropping_rect[1]:cropping_rect[3], cropping_rect[0]:cropping_rect[2]]

    # cv2.imshow("Cropped Image", image)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 4, 200)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)

    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    # draw the contours on a copy of the original image
    # cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1, x2, y2 = x, y, x + w, y + h
        boxes.append([x1, y1, x2, y2])

    nms_boxes = non_max_suppression_fast(np.array(boxes), 0.5)

    for box in nms_boxes:
        x1, y1, x2, y2 = box

        # Draw a bounding box
        # cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # print(len(contours), "objects were found in this image.")

    # cv2.namedWindow("Dilated image", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    #
    # cv2.imshow("Dilated image", dilate)
    # cv2.imshow("contours", image_copy)
    # cv2.waitKey(0)

    # Convert the nms_boxes to the original uncropped image coordinates
    boxes = []
    for box in nms_boxes:
        original_x1, original_y1, original_x2, original_y2 = transform_bbox_to_original(box, cropping_rect[0],
                                                                                        cropping_rect[1])
        boxes.append([original_x1, original_y1, original_x2, original_y2])

        # Draw the bounding boxes on the original image
        # original_image = cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # original_image = cv2.rectangle(original_image, (original_x1, original_y1), (original_x2, original_y2), (0, 0, 255), 2)

    # cv2.imshow("contours", original_image)
    # cv2.waitKey(0)

    return boxes, contours, cropping_rect


def find_contour_width(contour):
    grouped_coordinates = {}
    for x, y in contour:
        if y in grouped_coordinates:
            grouped_coordinates[y].append([x, y])
        else:
            grouped_coordinates[y] = [[x, y]]
    grouped_coordinates = list(grouped_coordinates.values())
    max_width = 0
    for grouped_coordinate in grouped_coordinates:
        # print(grouped_coordinate)
        min_x = 0
        max_x = 9999999
        for x, y in grouped_coordinate:
            if x < max_x:
                max_x = x
            if x > min_x:
                min_x = x
        width = abs(max_x - min_x)
        if width > max_width:
            max_width = width
    return max_width

def head_detection(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    l = []  # list to store each contour
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        l.append([i, area, c])

        # Ignore contours that are too small or too large (you can this piece of code when you have mutiple objects in the image)
        # if area < 10000 or   1763769 < area:
        # continue

    df = pd.DataFrame(l, columns=['index', 'Area', "contour"])
    df = df.sort_values("Area", ascending=False)
    df.reset_index(inplace=True)
    n = df["index"][0]

    # Get the largest contour
    largest_contour = contours[n]

    # Calculate the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the horizontal midpoint of the bounding rectangle
    midpoint_x = x + w // 2

    # Split the largest contour into upper and lower halves
    upper_half_contour = largest_contour[largest_contour[:, :, 1] <= y + h // 2]
    lower_half_contour = largest_contour[largest_contour[:, :, 1] > y + h // 2]

    upper_width = find_contour_width(upper_half_contour)
    lower_width = find_contour_width(lower_half_contour)

    # Draw the upper and lower half contours for visualization purposes
    temp = cv2.drawContours(src.copy(), [upper_half_contour], -1, (0, 0, 255), 1)
    temp = cv2.drawContours(temp, [lower_half_contour], -1, (0, 255, 0), 1)

    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    cv2.imshow("contours", temp)
    cv2.waitKey(1)

    # # Draw each contour only for visualisation purposes
    # temp = cv2.drawContours(src, contours, n, (0, 0, 255), 1)
    # cv2.imshow("contours", temp)
    # cv2.waitKey(1)

    if upper_width > lower_width:
        print("Head at the top")
        return True
    else:
        print("Head at the bottom")
        return False