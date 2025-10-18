import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------------------------------------------
# Load the trained emotion detection model
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("emotion_model.keras")  # make sure the file name matches
    return model

model = load_model()

# ------------------------------------------------------
# Emotion Labels (adjust according to your model)
# ------------------------------------------------------
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------------------------------------------
# Load Haar Cascade for face detection
# ------------------------------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.title("😊 Face Emotion Detection App")
st.write("Upload an image and let the AI detect the emotions in it.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected in the uploaded image.")
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            # Predict emotion
            preds = model.predict(roi_gray)[0]
            label = EMOTIONS[np.argmax(preds)]
            confidence = np.max(preds)

            # Draw bounding box and label
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_array, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display prediction
            st.subheader(f"Detected Emotion: **{label}** 😃")
            st.bar_chart(dict(zip(EMOTIONS, preds)))

        # Display final image
        st.image(img_array, caption="Emotion Detection Result", use_column_width=True)

else:
    st.info("👆 Please upload an image file to begin emotion detection.")
