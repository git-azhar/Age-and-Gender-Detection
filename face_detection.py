import cv2 as cv
import numpy as np


class FaceDetection:
    def __init__(self, face_network):
        self.face_net = face_network

    def get_face_boxes(self, image: np.ndarray, conf_threshold: float = 0.7) -> (np.ndarray, list):
        """
        Function to get bounding boxes of all detected faces in an image.
        :param image: image
        :param conf_threshold: confidence threshold of detection
        """

        image_frame = image.copy()
        frame_height = image_frame.shape[0]
        frame_width = image_frame.shape[1]

        blob = cv.dnn.blobFromImage(image_frame, scalefactor=1.0, size=(300, 300), mean=True, swapRB=False)

        self.face_net.setInput(blob)
        detections = self.face_net.forward()
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
