"""
This file contains the functions to send the data to AutoAI for annotation
"""
# importing libraries
import requests
import random
import shutil
import json
import random
import string
import math
import operator
from functools import reduce
import datetime
import numpy as np
import threading
import os

# importing user defined modules

CLASSES_US = ["green", "golden", "mech"]
AUTOAI_URL = "http://10.10.10.10/backend/resource/"
DETECTION_MODEL_ID = "64b536d86310da323b6b0266"
CLASSIFICATION_MODEL_ID = "64b536e46310da80366b02be"
TAG_PREFIX = "GoogleNext_"


def transform_bbox_to_original(cropped_bbox, crop_start_x, crop_start_y):
    # Unpack the cropped bounding box coordinates
    cropped_x1, cropped_y1 = cropped_bbox

    # Calculate the coordinates of the bounding box in the original image
    original_x1 = cropped_x1 + crop_start_x
    original_y1 = cropped_y1 + crop_start_y

    # Return the transformed bounding box coordinates
    return original_x1, original_y1


# Utility functions
def random_id(digit_count):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(digit_count))


highContrastingColors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)',
                         'rgba(227,0,255,1)']

annotation_id = {}
for index, object_class in enumerate(CLASSES_US):
    annotation_id[object_class] = index
annotation_id["screw"] = len(CLASSES_US)


# Function to send to AutoAI
def send_to_autoai(status, csv, model, label, tag, confidence_score, prediction, model_type,
                   filename, imageAnnotations, file_type="image/png", prompt=None):
    try:
        payload = {'status': status,
                   'csv': csv,
                   'model': model,
                   'label': label,
                   'tag': TAG_PREFIX + tag,
                   'confidence_score': confidence_score,
                   'prediction': prediction,
                   'imageAnnotations': imageAnnotations,
                   'model_type': model_type}

        if prompt is not None:
            payload['prompt'] = prompt

        files = [('resource', (filename, open(filename, 'rb'), file_type))]
        headers = {}
        response = requests.request(
            'POST', AUTOAI_URL, headers=headers, data=payload, files=files, verify=False)
        if response.status_code == 200:
            print('Successfully sent to AutoAI', end="\r")
            return True
        else:
            print('Error while sending to AutoAI')
            return False

    except Exception as e:
        print('Error while sending data to Auto AI : ', e)
        return False


# Function to create annotation in AutoAI format
def json_creater(inputs, closed):
    data = []
    for index, input in enumerate(inputs):
        # JSON Object for the metadata and vertices
        json_id = random_id(8)
        color = highContrastingColors[index % 5]
        sub_json_data = {}
        sub_json_data["id"] = json_id
        sub_json_data["name"] = json_id
        sub_json_data["color"] = color
        sub_json_data["isClosed"] = closed
        sub_json_data["selectedOptions"] = [{"id": "0", "value": "root"},
                                            {"id": annotation_id[inputs[input]], "value": inputs[input]}]

        points = eval(input)
        # points = np.fromstring(input, dtype=int)


        sorted_coords = points.copy()
        vertices = []
        is_first = True
        for vertex in sorted_coords:
            print(vertex)
            vertex_json = {}
            if is_first:
                vertex_json["id"] = json_id
                vertex_json["name"] = json_id
                is_first = False
            else:
                json_id = random_id(8)
                vertex_json["id"] = json_id
                vertex_json["name"] = json_id
            vertex_json["x"] = vertex[0]
            vertex_json["y"] = vertex[1]
            vertices.append(vertex_json)
        sub_json_data["vertices"] = vertices
        data.append(sub_json_data)
    return json.dumps(data)


def send_to_autoai_image(status, csv, model, label, tag, confidence_score, prediction, model_type,
                         filename, imageAnnotations, file_type, file_prefix):



    shutil.copy(filename, file_prefix + '_' + filename)
    filename = file_prefix + '_' + filename
    send_to_autoai(status, csv, model, label, tag, confidence_score, prediction, model_type,
                   filename, imageAnnotations, file_type, None)
    os.remove(filename)


def send_to_autoai_annotation(status, csv, model, label, tag, confidence_score, prediction, model_type,
                              filename, imageAnnotations, file_type, file_prefix,

                              boxes, labels, cropping_rect):
    li = {}
    try:
        boxes = boxes.cpu().tolist()
    except:
        pass

    for labeld, box in zip(labels, boxes):
        # coord = box
        # xmin, ymin, xmax, ymax = round(coord[0]), round(coord[1]), round(coord[2]), round(coord[3])
        # li[f"[[{xmin}, {ymin}], [{xmin}, {ymax}], [{xmax}, {ymax}], [{xmax}, {ymin}]]"] = labeld

        coord = box
        coord = coord.tolist()
        temp = []
        for i in coord:
            # temp.append(i[0])

            corrected_coords = transform_bbox_to_original(i[0], cropping_rect[0], cropping_rect[1])
            temp.append(corrected_coords)

        coord = temp
        li[str(coord)] = labeld


    annotations = json_creater(li, True)
    send_to_autoai(status=status,
                   csv=csv, # + "<br>Annotations : %s" % str(np.array(boxes, dtype=np.int)),
                   model=model,
                   label=label,
                   tag=tag,
                   confidence_score=confidence_score,
                   prediction=prediction,
                   model_type=model_type,
                   filename=filename,
                   imageAnnotations=annotations,
                   file_type=file_type,
                   prompt=None)
    # os.remove(filename)



def send_to_autoai_classes(status, csv, model, label, tag, confidence_score, prediction, model_type,
                   filename, imageAnnotations, file_type,

            classes, boxes, scores, files_crop, object_ids):

    for predicted_label, box, score, _ in zip(classes, boxes, scores, files_crop):
        csv = csv + "<br>%s : %s : %s" % (predicted_label, box, round(np.max(score) * 100, 2))

    for predicted_label, _, score, file, object_id in zip(classes, boxes, scores, files_crop, object_ids):
        new_csv = csv + "<br>Object ID : %s" % object_id

        send_to_autoai(status=status,
                       csv=new_csv,
                       model=model,
                       label=predicted_label,
                       tag=tag,
                       confidence_score=round(np.max(score) * 100, 2),
                       prediction=prediction,
                       model_type=model_type,
                       filename=file,
                       imageAnnotations="",
                       file_type=file_type,
                       prompt=None)
