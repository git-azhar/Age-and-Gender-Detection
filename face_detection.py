import cv2 as cv



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




# for bbox in bboxes:
#     face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
#            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
