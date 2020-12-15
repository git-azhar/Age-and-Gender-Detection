import cv2 as cv
import matplotlib.pyplot as plt

from src.face_detection import FaceDetection
from src.load_networks import LoadNetworks

padding = 20
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

frame = cv.imread('images/jolie.jpg')
load_networks = LoadNetworks()

# face-detection
face_net = load_networks.load_face_net()
detect_face = FaceDetection(face_net)
frame_face, bboxes = detect_face.get_face_boxes(image=frame)

plt.figure()
plt.imshow(frame_face)
plt.show()
if not bboxes:
    print('No face detected')
