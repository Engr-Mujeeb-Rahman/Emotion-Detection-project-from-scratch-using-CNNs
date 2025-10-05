# Emotion-Detection-project-from-scratch-using-CNNs

This project implements a **Convolutional Neural Network (CNN)** from scratch for **facial emotion recognition**.  
It can detect human emotions (e.g., Angry, Happy, Sad, Fear, Surprise, Neutral, etc.) from images or webcam input.

---

## 🚀 Features
- Built **without pre-trained models** – pure CNN architecture designed and trained from scratch.
- Trained on **FER-2013 dataset**.
- Link of Dataset **https://www.kaggle.com/datasets/msambare/fer2013**.
- Real-time detection using **OpenCV + TensorFlow/Keras**.
- Supports saving/loading model (`.keras` file).

---

## 🏗 Model Architecture
- **Input Layer**: Grayscale image (48x48 pixels).
- **Conv2D + MaxPooling2D layers** for feature extraction.
- **Dropout layers** for regularization.
- **Fully Connected Dense layers** for classification.
- **Output Layer**: Softmax activation for 7 emotion classes.

---

## 📂 Project Structure
EMOTION_DETECTION/
│── app.py
│── archive_4.zip (Datset)
│── emotion_detection_from_scrach_using_CNN....ipynb
│── emotion_model.keras
│── haarcascade_frontalface_default.xml
│── model.py
│── extracted_files/
│── .ipynb_checkpoints/
