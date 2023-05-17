from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder='.', static_folder = 'static')
# Load the trained model
bmi_model = tf.keras.models.load_model("densenet_model_checkpoint.h5")
gender_model = tf.keras.models.load_model("inception3_model_checkpoint.h5")
# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access the webcam
video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face_image = rgb_frame[y:y+h, x:x+w]

            # Preprocess the face image
            face_image = cv2.resize(face_image, (224, 224))  # Adjust the size as per your model's input requirements
            face_image = np.expand_dims(face_image, axis=0)
            face_image = face_image / 255.0  # Normalize the image

            # Predict BMI
            bmi_prediction = bmi_model.predict(face_image)
            bmi_value = bmi_prediction[0]  # Assuming your model predicts a single value
            #Predict Gender
            gender_prediction = gender_model.predict(face_image)
            gender_value = gender_prediction
            if gender_value <0.5:
                gender_value = 'Female'
            else:
                gender_value = 'Male'
            # Display the BMI and Gender predictions on the frame
            bmi_text = f'BMI: {float(bmi_value):.2f}'
            gender_text = f'Gender: {gender_value}'
            cv2.putText(frame, f'{bmi_text}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'{gender_text}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)