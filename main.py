import cv2 as cv
import matplotlib.pyplot as plt
from face_detection import FaceDetection


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']


frame = cv.imread('images/jolie.jpg')
detect_face = FaceDetection()
frame_face, bboxes = detect_face.get_face_boxes(image=frame)

plt.figure()
plt.imshow(frame_face)
plt.show()
if not bboxes:
    print('No face detected')
