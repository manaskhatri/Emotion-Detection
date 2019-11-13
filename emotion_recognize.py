'''
import face_recognition


image1 = face_recognition.load_image_file("manas.jpg")
image2 = face_recognition.load_image_file("neel.jpeg")
face_locations = face_recognition.face_locations(image1)
top, right, bottom, left = face_locations[0]
face_image = image1[top:bottom, left:right]

#image1 = face_recognition.load_image_file("../test_images/index1.jpg")
#image2 = face_recognition.load_image_file("../test_images/index2.jpeg")
encoding_1 = face_recognition.face_encodings(image1)[0]
encoding_2 = face_recognition.face_encodings(image2)[0]
results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)

'''
import cv2
import numpy as np
from keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array



face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img):
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    allfaces = []   
    rects = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x,w,y,h))
    return rects, allfaces, img


emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
img = cv2.imread("hf.jpg")
#face_image  = cv2.imread("hf.jpg")
img = cv2.resize(img, (48,48))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.reshape(img, [1, img.shape[0], img.shape[1], 1])
model = load_model("model_v6_23.hdf5")
predicted_class = np.argmax(model.predict(img))
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
print(predicted_label)

img = cv2.imread("hf.jpg")
rects, faces, image = face_detector(img)
i = 0
for face in faces:
    roi = face.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    # make a prediction on the ROI, then lookup the class
    preds = model.predict(roi)[0]
    label = label_map[predicted_class]   

    #Overlay our detected emotion on our pic
    label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
    i =+ 1
    cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    
cv2.imshow("Emotion Detector", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
