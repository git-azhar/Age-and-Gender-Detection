import cv2 as cv
import matplotlib.pyplot as plt

from age_gender import AgeGender
from face_detection import FaceDetection
from load_networks import LoadNetworks

padding = 20
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

frame = cv.imread('data/img2.jpg')
load_networks = LoadNetworks()

# Load Networks
face_net = load_networks.load_face_net()
gender_net = load_networks.load_gender_net()
age_net = load_networks.load_age_net()

# face-detection
detect_face = FaceDetection(face_net)
frame_face, bboxes = detect_face.get_face_boxes(image=frame)

# Age and Gender detection
age_gender_detection = AgeGender(age_network=age_net, gender_network=gender_net, image_frame=frame_face, bboxes=bboxes)
gender, age = age_gender_detection.detect_age_gender()

