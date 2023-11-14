import os
#import utils_vision
#import cv2
import random
import datetime
import numpy as np

import autoai_utils

input_folder_path = "C:/Users/Sai Siddhartha/Downloads/send_screw_to_autoai/send_screw_to_autoai/train/scissors"
files = os.listdir(input_folder_path)

DETECTION_MODEL_ID = "6549fedd4f482a74822f6105"

for index, file in enumerate(files):
    request_id = random.randint(1, 9999999)
    datetime_date = datetime.datetime.now().strftime("%d-%m-%Y")
    datetime_time = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    print("Processing file %s/%s" % (index, len(files)))
    file_path = os.path.join(input_folder_path, file)

    '''color_image = cv2.imread(file_path)

    boxes, contours, cropping_rect = utils_vision.countour_detect_screws(color_image)

    for coords in boxes:

        color_image = cv2.rectangle(color_image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)


    cv2.imshow("color_image", color_image)
    cv2.waitKey(1)'''


    # Send the results of the detection model to RLEF
    csv = "ID : %s<br>" % request_id

    autoai_utils.send_to_autoai("backlog",
                                           csv,
                                           DETECTION_MODEL_ID,
                                           "watch",
                                           "DETECTION_%s" % datetime_date,
                                           100,
                                           "predicted",
                                           "segmentation",
                                           file_path,
                                           "",
                                           "image/png",
                                           "")


