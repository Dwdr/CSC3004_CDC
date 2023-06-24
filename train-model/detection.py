from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import keras
import threading

app = Flask(__name__)

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = keras.models.load_model("lstm-model.h5")

lm_list = []
label = ""

@app.route('/')
def index():
    return render_template('index.html')

def make_landmark_timestep(results):
    # Function to convert pose landmarks to a flattened list
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    # Function to draw landmarks on the frame
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    # Function to draw the predicted class label on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    if label == "Violent":
        fontColor = (0, 0, 255)
    else:
        fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label), bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

def detect(model, lm_list):
    # Function to perform violence detection using the LSTM model
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "Violent"
    else:
        label = "Neutral"
    return str(label)

def generate_frames():
    # Generator function to process video frames and yield as MJPEG frames
    global lm_list, label
    i = 0
    warm_up_frames = 60
    while True:
        ret, frame = cap.read()
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        i = i + 1
        if i > warm_up_frames:
            print("Start detecting...")
            if results.pose_landmarks:
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
                if len(lm_list) == 20:
                    t1 = threading.Thread(target=detect, args=(model, lm_list,))
                    t1.start()
                    lm_list = []
                x_coordinate = []
                y_coordinate = []
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_coordinate.append(cx)
                    y_coordinate.append(cy)
                if label == "Neutral":
                    cv2.rectangle(img=frame,
                                  pt1=(min(x_coordinate), max(y_coordinate)),
                                  pt2=(max(x_coordinate), min(y_coordinate) - 25),
                                  color=(0, 255, 0),
                                  thickness=1)
                elif label == "Violent":
                    cv2.rectangle(img=frame,
                                  pt1=(min(x_coordinate), max(y_coordinate)),
                                  pt2=(max(x_coordinate), min(y_coordinate) - 25),
                                  color=(0, 0, 255),
                                  thickness=3)

                frame = draw_landmark_on_image(mpDraw, results, frame)
        frame = draw_class_on_image(label, frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Route to provide the video feed as a response with MJPEG content type
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
