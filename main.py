from face_detection import Detection
import cv2 as cv
img_path = 'data/img4.jpg'
detection = Detection()
img = detection.detect_age_gender(image_path=img_path)

cv.imshow("Age Gender Demo", img)
# cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
cv.waitKey(0)





