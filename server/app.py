import base64
import datetime
import os
import smtplib
import ssl
import threading
import io
from email.mime.text import MIMEText

import boto3
import cv2
import numpy as np

import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import keras
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)
load_dotenv()  # Load environment variables

connected_devices = []
landmark_list = []

# Load the model
model = keras.models.load_model("lstm-model.h5")

# Initialise pose detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Configure AWS access keys and region
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = os.getenv("AWS_REGION")
port = os.getenv("PORT")

# Specify the S3 bucket name and target path
s3 = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region_name,
)
video_bucket_name = os.getenv("AWS_VIDEO_BUCKET_NAME")
video_bucket_prefix = os.getenv("AWS_VIDEO_BUCKET_PREFIX")
image_bucket_name = os.getenv("AWS_IMAGE_BUCKET_NAME")
image_bucket_prefix = os.getenv("AWS_IMAGE_BUCKET_PREFIX")
analysis_bucket_name = os.getenv("AWS_ANALYSIS_BUCKET_NAME")
analysis_bucket_prefix = os.getenv("AWS_ANALYSIS_BUCKET_PREFIX")

from_email = os.getenv("EMAIL_ADDRESS")
password = os.getenv("EMAIL_PASSWORD")
smtp = os.getenv("SMTP_SERVER")


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


"""This function saves the image frame into an 
   Amazon S3 bucket"""


def save_image(image_data, uid, movement_value):
    # Split the data URL to extract the base64 image data
    image_data_parts = image_data.split(",", 1)
    base64_image_data = image_data_parts[1]

    # Convert the base64 image data from string to bytes
    image_bytes = base64.b64decode(base64_image_data)

    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"image_{uid}_{timestamp}.png"

    # Crime has been detected
    if movement_value == 1:
        key = f"images/crime/{filename}"
        s3.put_object(
            Bucket=image_bucket_name, Key=key, Body=image_bytes, ContentType="image/png"
        )
    else:
        key = f"images/no_crime/{filename}"
        # Upload image to the S3 bucket
        s3.put_object(
            Bucket=image_bucket_name, Key=key, Body=image_bytes, ContentType="image/png"
        )


"""This function is used to load the names of all the images stored in the S3 bucket.
   It returns a list of file names."""


def load_cloud_images():
    try:
        # Initialize an empty list to store the file names
        file_names = []

        # List objects in the 'cdcimagebucket' with the specified prefix
        crime_response = s3.list_objects_v2(
            Bucket=image_bucket_name, Prefix=image_bucket_prefix + "crime/"
        )

        # Iterate over the objects and add the file names to the list
        for obj in crime_response.get("Contents", []):
            key = obj["Key"]
            # Add the file name to the list
            file_names.append(key)

        no_crime_response = s3.list_objects_v2(
            Bucket=image_bucket_name, Prefix=image_bucket_prefix + "no_crime/"
        )

        for obj in no_crime_response.get("Contents", []):
            key = obj["Key"]
            file_names.append(key)

        # Return the list of file names
        return file_names

    except Exception as e:
        print(f"Error loading images from S3: {e}")
        return []


"""This function is used to send a notification to a user if a crime has been detected.
   It takes in the user's email address as a parameter."""


def send_email(to_email):
    # Compose the email content
    subject = "Crime Detected"
    body = "A crime has been detected."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        port = 465
        context = ssl.create_default_context()
        # modify the smtp server according to your email provider
        with smtplib.SMTP_SSL(smtp, port, context=context) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
            server.quit()

        print("Email sent successfully")
    except Exception as e:
        print("Error sending email:", str(e))


"""This function is used to send a welcome email to a new user.
   It takes in the user's email address as a parameter."""


def send_welcome_email(to_email):
    subject = "Welcome to the System"
    body = "Welcome to our system! Thank you for joining."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        port = 465
        context = ssl.create_default_context()
        # modify the smtp server according to your email provider
        with smtplib.SMTP_SSL(smtp, port, context=context) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
            server.quit()

        print("Welcome email sent successfully")
    except Exception as e:
        print("Error sending welcome email:", str(e))


def add_padding(image_data):
    padding = 4 - (len(image_data) % 4)
    image_data += "=" * padding
    return image_data


def detect(model, client_id, lm_list):
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "Violent"
    else:
        label = "Neutral"

    if label == "Violent":
        print("Crime detected")
        has_crime = 1
        for device in connected_devices:
            if device["client_id"] == client_id:
                device["has_crime"] = has_crime  # Update the 'has_crime' value
                client_email = device["client_email"]
                if client_email:
                    send_email(client_email)
                break
    else:
        has_crime = 0
        # Check if the client_id already exists in the connected_devices list
        existing_device = next(
            (
                device
                for device in connected_devices
                if device["client_id"] == client_id
            ),
            None,
        )

        if existing_device:
            existing_device["has_crime"] = has_crime  # Update the 'has_crime' value
        else:
            # Add the device to the connected_devices list
            connected_devices.append(
                {
                    "client_id": client_id,
                    "has_crime": has_crime,
                    "client_email": "",
                }
            )
    socketio.emit("detection_result", {"client_id": client_id, "has_crime": has_crime})


"""This function is used to detect crime in an image.
   It takes in the image data as a base64-encoded string and returns True if the image contains violence, False otherwise."""


def detect_crime(image_data):
    print("Decoding image data...")
    try:
        # Load the pre-trained violence detection model
        sgd = tf.keras.optimizers.legacy.SGD(
            learning_rate=0.01, momentum=0.9, nesterov=True
        )

        model = load_model("modelnew.h5")
        model.compile(
            optimizer=tf.keras.optimizers.legacy.SGD(
                learning_rate=0.01, momentum=0.9, nesterov=True
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Convert the image data to a NumPy array
        image = np.frombuffer(base64.b64decode(image_data.split(",")[1]), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        # Resize the image to the expected shape
        image = cv2.resize(image, (128, 128))

        # Convert the grayscale image to a 3-channel image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Expand dimensions to match the input shape of the model
        image = np.expand_dims(image, axis=0)

        # Predict whether the image contains violence
        prediction = model.predict(image)

        # Get the predicted class index
        predicted_class = np.argmax(prediction, axis=1)[0]

        print("Prediction:", prediction)
        print("Predicted class:", predicted_class)

        # Return True if the predicted class is 1 (violence), False otherwise
        return int(predicted_class)
    except Exception as e:
        print("Error during image processing:", str(e))
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/file-names", methods=["GET"])
def file_names():
    return {"fileNames": load_cloud_images()}


# This function is called when a client connects to the server
@socketio.on("connect")
def handle_connect():
    socketio.emit("detection_result", {"status": "1"})
    print("Client connected")


# This function is called when a client disconnects from the server
@socketio.on("disconnect")
def disconnect(client_uid):
    device_to_remove = None

    for device in connected_devices:
        if device.get("client_id") == client_uid:
            device_to_remove = device
            break

    if device_to_remove:
        connected_devices.remove(device_to_remove)
        print(f"Client {client_uid} disconnected")


"""Handles the detection of crime in a video frame on a separate thread"""


def detect_frame_worker(client_id, image_data):
    print("\nDetecting crime for client:", client_id, "\n")

    has_crime = int(detect_crime(image_data))  # Convert boolean to integer

    if has_crime == 1:
        save_image(image_data, client_id, 1)
        print("Crime detected")
        for device in connected_devices:
            if device["client_id"] == client_id:
                device["has_crime"] = has_crime  # Update the 'has_crime' value
                client_email = device["client_email"]
                if client_email:
                    send_email(client_email)
                break
    else:
        save_image(image_data, client_id, 0)
        # Check if the client_id already exists in the connected_devices list
        existing_device = next(
            (
                device
                for device in connected_devices
                if device["client_id"] == client_id
            ),
            None,
        )

        if existing_device:
            existing_device["has_crime"] = has_crime  # Update the 'has_crime' value
        else:
            # Add the device to the connected_devices list
            connected_devices.append(
                {
                    "client_id": client_id,
                    "has_crime": has_crime,
                    "client_email": "",
                }
            )

    print(connected_devices)

    socketio.emit("detection_result", {"client_id": client_id, "has_crime": has_crime})


""" This function is called when a client sends a detect-frame request to the server. Since the client is sending a POST request, we need to handle the OPTIONS request first.
    The detect-frame request contains the client's unique ID and the image data. We create a new thread to process the image data and detect crime."""


@app.route("/detect-frame", methods=["POST", "OPTIONS"])
def handle_detect_frame():
    if request.method == "OPTIONS":
        # Respond to the preflight request
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    try:
        image_data = request.json["image"]
        uid_data = request.json["uid"]
        thread = threading.Thread(
            target=detect_frame_worker,
            args=(uid_data, image_data),
        )
        thread.start()
        print("Thread started for client:", uid_data)
    except Exception as e:
        print("Error handling detect-frame request:", str(e))
        socketio.emit("detection_result", {"status": "0"})

    return jsonify({}), 200


@app.route("/collect-frames", methods=["POST", "OPTIONS"])
def handle_collect_frames():
    global landmark_list
    if request.method == "OPTIONS":
        # Respond to the preflight request
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    try:
        uid_data = request.json["uid"]
        image_data = request.json["image"]
        image_data_cleaning = image_data.split(",")[1]
        padded_image_data = add_padding(image_data_cleaning)
        decoded_image_data = base64.urlsafe_b64decode(padded_image_data)
        try:
            image = Image.open(io.BytesIO(decoded_image_data), formats=["png"])
            image_np = np.array(image)
        except (IOError, OSError) as e:
            print("Error: Failed to open the image:", str(e))

        frameRGB = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            landmark_list.append(lm)
            print("Landmark list length:", len(landmark_list))

            if len(landmark_list) == 20:
                t1 = threading.Thread(
                    target=detect,
                    args=(
                        model,
                        uid_data,
                        landmark_list,
                    ),
                )
                t1.start()
                landmark_list = []
                print("Thread started for client:", uid_data)

    except Exception as e:
        print("Error handlingcollect-frame request:", str(e))
        socketio.emit("detection_result", {"status": "0"})

    return jsonify({}), 200


"""This function is called when a client sends a get-connected-devices request to the server. It returns a list of all the connected devices."""


@app.route("/connected-devices", methods=["GET"])
def get_connected_devices():
    devices = []
    for device in connected_devices:
        client_id = device["client_id"]
        has_crime = device["has_crime"]
        client_email = device["client_email"]
        devices.append(
            {
                "client_id": client_id,
                "has_crime": has_crime,
                "client_email": client_email,
            }
        )
    return jsonify(devices)


"""This function is called when a client sends an add-email request to the server. It adds the client's email address to the connected_devices list."""


@app.route("/add-email", methods=["POST"])
def add_email():
    email = request.json["email"]
    uid = request.json["uid"]
    found = False

    for device in connected_devices:
        if device["client_id"] == uid:
            device["client_email"] = email
            found = True
            break

    if not found:
        connected_devices.append(
            {"client_id": uid, "client_email": email, "has_crime": 0}
        )

    send_welcome_email(email)
    print(connected_devices)

    return jsonify({}), 200


"""This function is called when a client sends a remove-email request to the server. It removes the client's email address from the connected_devices list."""


@app.route("/remove-email", methods=["POST"])
def remove_email():
    email = request.json["email"]
    uid = request.json["uid"]
    for device in connected_devices:
        if device["client_id"] == uid and device["client_email"] == email:
            device["client_email"] = ""
            return jsonify({}), 200
    return jsonify({"error": "Email not found for the specified client ID."}), 404


@app.before_request
def before_request():
    if request.method == "POST" and request.path == "/detect-frame":
        request.environ["CONTENT_TYPE"] = "application/json"


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
