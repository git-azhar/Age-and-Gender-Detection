# Age and Gender Detection (Opencv)

## Models

You can download caffe-models here:
- [Gender Net](https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0)
- [Age net](https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0)

Here are [OpenCV pre-trained models](https://github.com/opencv/opencv/tree/master/data/haarcascades)

## Getting Started:
Here's how I'm appraochin gthis problem:
- Gender detection as a classification problem
- Age detection problem as a regression problem.

However, accurate estimation of age is not an easy task. so instead of giving a precise number I'm going to detect the age range.


## Face Detection:
First task is to detect faces in a image as without detecting faces we won't be able to move further to the task of gender, age detection.

to get bounding boxes use follwoing method.
```
detect_face = FaceDetection(face_net)
frame_face, bboxes = detect_face.get_face_boxes(image=frame)
```
