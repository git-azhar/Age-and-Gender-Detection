import cv2 as cv


class FaceDetection:
    def __init__(self):
        self.age_proto = "age_deploy.prototxt"
        self.age_model = "models/age_net.caffemodel"

        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "models/gender_net.caffemodel"

        self.face_proto = "opencv_face_detector.pbtxt"
        self.face_model = "opencv_face_detector_uint8.pb"

    @classmethod
    def load_network(cls, model_path, network_file):
        # Load network
        network = cv.dnn.readNet(model_path, network_file)
        return network

    def get_face_boxes(self, image, conf_threshold: float = 0.7):
        net = self.load_network(model_path=self.face_model, network_file=self.face_proto)
        image_frame = image.copy()
        frame_height = image_frame.shape[0]
        frame_width = image_frame.shape[1]

        blob = cv.dnn.blobFromImage(image_frame, scalefactor=1.0, size=(300, 300), mean=True, swapRB=False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                bboxes.append([x1, y1, x2, y2])

                cv.rectangle(img=image_frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),
                             thickness=5, lineType=8)
        return image_frame, bboxes

    # for bbox in bboxes:
    #     face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
    #            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
