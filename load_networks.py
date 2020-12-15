import cv2 as cv


class LoadNetworks:
    def __init__(self):
        self.face_proto = "opencv_face_detector.pbtxt"
        self.face_model = "opencv_face_detector_uint8.pb"

        self.age_proto = "age_deploy.prototxt"
        self.age_model = "models/age_net.caffemodel"

        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "models/gender_net.caffemodel"

    def load_age_net(self) -> cv.dnn_Net:
        # Load network
        return cv.dnn.readNet(self.age_model, self.age_proto)

    def load_face_net(self) -> cv.dnn_Net:
        # Load network
        return cv.dnn.readNet(self.face_model, self.face_proto)

    def load_gender_net(self) -> cv.dnn_Net:
        # Load network
        return cv.dnn.readNet(self.gender_model, self.gender_proto)
