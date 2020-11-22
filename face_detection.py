import cv2 as cv
import matplotlib.pyplot as plt


def get_face_boxes(net, image, conf_threshold: float = 0.7):
    frame_OpencvDnn = image.copy()
    frame_height = frame_OpencvDnn.shape[0]
    frame_width = frame_OpencvDnn.shape[1]

    blob = cv.dnn.blobFromImage(frame_OpencvDnn, scalefactor=1.0, size=(300, 300), mean=True, swapRB=False)

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

            cv.rectangle(img=frame_OpencvDnn, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),
                         thickness=int(round(frame_height / 150)), lineType=8)
    return frame_OpencvDnn, bboxes


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

# for bbox in bboxes:
#     face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
#            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
