from __future__ import division
import dlib
import cv2
import keras
import numpy as np
import collections
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    transfer_model = load_model('modeltransfer1.h5')

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

'''
	("mouth", (48, 68)),
	("right eyebrow", (17, 22)),
	("left eyebrow", (22, 27)),
	("right eye", (36, 42)),
	("left eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
'''

def image_prep(x):
    #img = image.load_img(x, target_size= (224,224))
    img_array = image.img_to_array(x)
    img = img_array.astype("float32")
    img_dims = np.expand_dims(img, axis = 0)
    img = keras.applications.mobilenet.preprocess_input(img_dims)
    return img

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

camera = cv2.VideoCapture(0)

preds = {'0':'jawline', '1':'nose', '2':'lefteye', '3': 'righteye', '4': 'mouth', '5':'fullface'}

X = 0
Y = 0
W = 350
H = 350

while True:

    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)
    dets = detector(frame_resized, 1)
    #print(len(dets))
    #print(dets)
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            #print(shape)
            shape = shape_to_np(shape)

    roi = frame[X:X+W, Y:Y+W]
    img1 = cv2.resize(roi, (224,224))
    prep_image = image_prep(img1)
    prediction = transfer_model.predict(prep_image)
    prediction = np.argmax(prediction)
    #print(prediction1)
    cv2.rectangle(frame, (X, Y), (X + W, Y + H), (0, 0, 0), 2)
    cv2.putText(frame, "Prediction: " + str(prediction), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            #print(shape)
            shape = shape_to_np(shape)
            if prediction == 0:
                cv2.putText(frame, "You choose jawline", (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                for (x, y) in shape[0:17]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 5, (0, 0, 255), -1)
                    #cv2.line(frame, (x, y), (x+1, y+1), (0,255,0), 0)

            if prediction == 1:
                cv2.putText(frame, "You choose nose", (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                for (x, y) in shape[27:35]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 5, (0, 0, 255), -1)

            if prediction == 2:
                cv2.putText(frame, "You choose lefteye", (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                for (x, y) in shape[42:48]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 5, (0, 0, 255), -1)

            if prediction == 3:
                cv2.putText(frame, "You choose righteye", (10, 390),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                for (x, y) in shape[36:42]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 5, (0, 0, 255), -1)

            if prediction == 4:
                cv2.putText(frame, "You choose mouth", (10, 390),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                for (x, y) in shape[48:68]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 5, (0, 0, 255), -1)

            if prediction == 5:
                cv2.putText(frame, "You choose fullface", (10, 390),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                for (x, y) in shape[0:17]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 5, (128, 128, 128), -1)
                for (x, y) in shape[48:68]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 4, (0, 0, 255), -1)
                for (x, y) in shape[27:35]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 4, (0, 145, 255), -1)
                for (x, y) in shape[36:48]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 4, (0, 255, 255), -1)
                for (x, y) in shape[17:27]:
                    cv2.circle(frame, (int(x/ratio), int(y/ratio)), 4, (255, 255, 255), -1)

    frame1 = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
    cv2.imshow("image", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
