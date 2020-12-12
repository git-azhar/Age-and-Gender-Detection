
import cv2 as cv
import matplotlib.pyplot as plt
from face_detection import get_face_boxes

face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"

age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load network
age_net = cv.dnn.readNet(age_model, age_proto)
gender_net = cv.dnn.readNet(gender_model, gender_proto)
face_net = cv.dnn.readNet(face_model, face_proto)

frame = cv.imread('images/jolie.jpg')

frame_face, bboxes = get_face_boxes(net=face_net, image=frame)

plt.figure()
plt.imshow(frame_face)
plt.show()
if not bboxes:
    print('No face detected')