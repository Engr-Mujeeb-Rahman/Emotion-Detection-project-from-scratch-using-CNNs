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
│── archive_4.zip # Dataset (FER-2013 or custom)<br>
│── emotion_detection_from_scratch_using_CNN.ipynb # Jupyter notebook (training)
│── emotion_model.keras # Saved CNN model
│── haarcascade_frontalface_default.xml # Haar Cascade for face detection
│── model.py # CNN model architecture
│── extracted_files/ # Extracted dataset files
│── .ipynb_checkpoints/ # Jupyter checkpoints
