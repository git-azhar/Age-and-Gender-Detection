import numpy as np
import cv2 as cv


class AgeGender:
    def __init__(self, age_network, gender_network, image_frame: np.ndarray, bboxes: list):
        self.img = image_frame
        self.bounding_boxes = bboxes
        self.age_net = age_network
        self.gender_net = gender_network
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        self.padding = 20
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def detect_age_gender(self) -> (str, (int, int)):
        """
        Function to detect Gender and Age
        :return: gender, Age
        """
        for bbox in self.bounding_boxes:
            face = self.img[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, self.img.shape[0] - 1),
                   max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, self.img.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, gender_preds[0].max()))

            # age detection
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            print("Age : {}, conf = {:.3f}".format(age, age_preds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(self.img, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2, cv.LINE_AA)

            cv.imshow("Age Gender Demo", self.img)
            # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
            cv.waitKey(0)
            return gender, age
