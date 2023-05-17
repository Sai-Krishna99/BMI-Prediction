import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Load the trained models
bmi_model = tf.keras.models.load_model("densenet_model_checkpoint.h5")
gender_model = tf.keras.models.load_model('inception3_model_checkpoint.h5')
# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access the webcam
video_capture = cv2.VideoCapture(0)

def predict_bmi_and_gender(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = rgb_frame[y:y+h, x:x+w]

        # Preprocess the face image
        face_image = cv2.resize(face_image, (224, 224))
        face_image = np.expand_dims(face_image, axis=0)
        face_image = face_image / 255.0

        # Predict BMI
        bmi_prediction = bmi_model.predict(face_image)
        bmi_value = bmi_prediction[0]

        # Predict Gender
        gender_prediction = gender_model.predict(face_image)
        gender_value = 'Female' if gender_prediction < 0.5 else 'Male'

        # Draw bounding box and text on the frame
        cv2.putText(frame, f'BMI: {float(bmi_value):.2f}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {gender_value}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

def main():
    st.title("BMI Prediction")

    # Start the video feed
    video_player = st.empty()

    while True:
        ret, frame = video_capture.read()
        processed_frame = predict_bmi_and_gender(frame)

        # Display the processed frame
        video_player.image(processed_frame, channels="RGB")

if __name__ == '__main__':
    main()
