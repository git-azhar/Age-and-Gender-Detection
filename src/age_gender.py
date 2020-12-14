import numpy as np
import cv2 as cv


class AgeGender:
    def __init__(self, age_network, gender_network, image_frame: np.ndarray, bboxes: list):
        self.img = image_frame
        self.bounding_boxes = bboxes
        self.padding = 20
        self.age_net = age_network
        self.gender_net = gender_network
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def detect_age(self):
        for bbox in self.bounding_boxes:
            face = self.img[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, self.img.shape[0] - 1),
                   max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, self.img.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)



