import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = load_model('emotion_model.keras')

classes = ['Angry' , 'Disgust' , 'Fear' , 'Happy' , 'Neutral' , 'Sad' , 'Surprise']

def process_face(face_img):
    face = cv2.resize(face_img , (48,48))
    face = face / 255
    face = np.expand_dims(face , axis= -1)
    face = np.expand_dims(face , axis= 0)
    return face

def predict_emotion(face_img):
    processed = process_face(face_img)
    prediction = model.predict(processed , verbose=0)
    class_idx = np.argmax(prediction)
    return classes[class_idx]



cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray , minNeighbors=10)

    for (x,y,w,h) in face:
        face_gray = gray[y:y+h , x:x+w]
        label = predict_emotion(face_gray)

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    cv2.imshow("Web cap face emotion detection" , frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()