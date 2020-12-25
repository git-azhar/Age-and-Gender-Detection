import time

import cv2 as cv
import numpy as np


class Detection:
    def __init__(self):

        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        self.padding = 20
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        self.face_proto = "opencv_face_detector.pbtxt"
        self.face_model = "opencv_face_detector_uint8.pb"

        self.age_proto = "age_deploy.prototxt"
        self.age_model = "models/age_net.caffemodel"

        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "models/gender_net.caffemodel"

    def get_face_boxes(self, image: np.ndarray, conf_threshold: float = 0.7) -> (np.ndarray, list):
        """
        Function to get bounding boxes of all detected faces in an image.
        :param image: image
        :param conf_threshold: confidence threshold of detection
        """

        image_frame = image.copy()
        frame_height = image_frame.shape[0]
        frame_width = image_frame.shape[1]

        face_net = self.load_face_net()
        blob = cv.dnn.blobFromImage(image_frame, scalefactor=1.0, size=(300, 300), mean=True, swapRB=False)
        face_net.setInput(blob)
        detections = face_net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                bboxes.append([x1, y1, x2, y2])

                cv.rectangle(img=image_frame, pt1=(x1, y1), pt2=(x2, y2), color=(200, 255, 0),
                             thickness=2, lineType=8)
        if not bboxes:
            print('No face detected')
        return image_frame, bboxes

    def detect_age_gender(self, image_path: str) -> np.ndarray:
        """
        Function to detect Gender and Age
        :return: gender, Age
        """

        frame = cv.imread(image_path)
        img, bboxes = self.get_face_boxes(image=frame.copy())
        for bbox in bboxes:
            face = img[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, img.shape[0] - 1),
                   max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, img.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            gender_net = self.load_gender_net()
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, gender_preds[0].max()))

            # age detection
            age_net = self.load_age_net()
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            print("Age : {}, conf = {:.3f}".format(age, age_preds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(img, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2, cv.LINE_AA)
        return img

    def load_age_net(self) -> cv.dnn_Net:
        # Load network
        return cv.dnn.readNet(self.age_model, self.age_proto)

    def load_face_net(self) -> cv.dnn_Net:
        # Load network
        return cv.dnn.readNet(self.face_model, self.face_proto)

    def load_gender_net(self) -> cv.dnn_Net:
        # Load network
        return cv.dnn.readNet(self.gender_model, self.gender_proto)
