from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder='.', static_folder='static')

# Load the trained models
bmi_model = tf.keras.models.load_model("densenet_model_checkpoint.h5")
gender_model = tf.keras.models.load_model('inception3_model_checkpoint.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = None
video_feed_running = False

def generate_frames():
    global video_capture

    while video_feed_running:
        if video_capture is not None:
            # Capture frame-by-frame from the webcam
            ret, frame = video_capture.read()

            if ret:
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame as an HTTP response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    global video_feed_running

    if not video_feed_running:
        return Response('', mimetype='multipart/x-mixed-replace; boundary=frame')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_feed')
def start_feed():
    global video_capture, video_feed_running

    if video_feed_running:
        return jsonify(message='Video feed already running')

    video_capture = cv2.VideoCapture(0)
    video_feed_running = True
    return jsonify(message='Video feed started')

@app.route('/stop_feed')
def stop_feed():
    global video_capture, video_feed_running

    if not video_feed_running:
        return jsonify(message='Video feed already stopped')

    video_capture.release()
    video_capture = None
    video_feed_running = False
    return jsonify(message='Video feed stopped')

@app.route('/predict')
def predict():
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()

    if not ret:
        return jsonify(predictions=[])

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    predictions = []
    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = rgb_frame[y:y+h, x:x+w]

        # Preprocess the face image
        face_image = cv2.resize(face_image, (224, 224))
        face_image = np.expand_dims(face_image, axis= 0)
        face_image = face_image / 255.0

        # Predict BMI
        bmi_prediction = bmi_model.predict(face_image)
        bmi_value = bmi_prediction[0]

        # Predict Gender
        gender_prediction = gender_model.predict(face_image)
        gender_value = 'Female' if gender_prediction < 0.5 else 'Male'

        predictions.append({'bmi': float(bmi_value), 'gender': gender_value})

    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
