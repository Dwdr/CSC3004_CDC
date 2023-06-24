"""
app.py

This is the main Flask application file for the client web application.
"""

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

from PIL import Image
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO
import keras
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)
load_dotenv()

connected_devices = []
landmark_lists = {}

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

# Specify the Email address, password and SMTP server
from_email = os.getenv("EMAIL_ADDRESS")
password = os.getenv("EMAIL_PASSWORD")
smtp = os.getenv("SMTP_SERVER")


def make_landmark_timestep(results):
    """
    Converts pose landmark data from the results object to a list of coordinates.

    Args:
        results: An object containing pose landmark data.

    Returns:
        A list of coordinates representing the pose landmarks.
        Each coordinate consists of the x, y, z, and visibility values of a landmark.
    """
    c_lm = []
    for id, landmark in enumerate(results.pose_landmarks.landmark):
        c_lm.append(landmark.x)
        c_lm.append(landmark.y)
        c_lm.append(landmark.z)
        c_lm.append(landmark.visibility)
    return c_lm


def save_image(image_data, uid, movement_value):
    """
    Saves an image to an S3 bucket based on the provided image data and movement value.

    Args:
        image_data (str): The data URL of the image in base64 format.
        uid (str): The unique identifier associated with the image.
        movement_value (int): The value indicating the presence of movement
        (1 if movement detected, else 0).

    Raises:
        ValueError: If the image data is not in a valid format or if the movement value is invalid.

    Returns:
        None
    """

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


def load_cloud_images():
    """
    Loads file names of images from an S3 bucket.

    Returns:
        list: A list of file names of images stored in the S3 bucket.

    Raises:
        None
    """
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

    except Exception as error:
        print(f"Error loading images from S3: {error}")
        return []


def send_email(to_email):
    """
    Sends an email notification.

    Args:
        to_email (str): The email address of the recipient.

    Returns:
        None

    Raises:
        None
    """
    # Compose the email content
    subject = "Crime Detected"
    body = "A crime has been detected."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        email_port = 465
        context = ssl.create_default_context()
        # modify the SMTP server according to your email provider
        with smtplib.SMTP_SSL(smtp, email_port, context=context) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
            server.quit()

        print("Email sent successfully")
    except Exception as error:
        print("Error sending email:", str(error))


def send_welcome_email(to_email):
    """
    Sends a welcome email to a new user.

    Args:
        to_email (str): The email address of the recipient.

    Returns:
        None

    Raises:
        None
    """
    subject = "Welcome to the System"
    body = "Welcome to our system! Thank you for joining."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        email_port = 465
        context = ssl.create_default_context()
        # modify the smtp server according to your email provider
        with smtplib.SMTP_SSL(smtp, email_port, context=context) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
            server.quit()

        print("Welcome email sent successfully")
    except Exception as error:
        print("Error sending welcome email:", str(error))


def add_padding(image_data):
    """
    Adds padding to the image data string to ensure it has a length that is a multiple of 4.

    Args:
        image_data (str): The image data string.

    Returns:
        str: The image data string with added padding.

    Raises:
        None
    """
    padding = 4 - (len(image_data) % 4)
    image_data += "=" * padding
    return image_data


def detect(model_file, client_id, lm_list):
    """
    Performs crime detection based on landmark data using a given model.

    Args:
        model: The crime detection model.
        client_id: The client ID associated with the landmark data.
        lm_list: A list of landmark data.

    Returns:
        None

    Raises:
        None
    """
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model_file.predict(lm_list)
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


@socketio.on("connect")
def handle_connect():
    """
    Event handler for client connection.

    Sends a detection result message with status "1" to the client and prints
    a log message indicating client connection.

    Returns:
        None

    Raises:
        None
    """
    socketio.emit("detection_result", {"status": "1"})
    print("Client connected")


@socketio.on("disconnect")
def disconnect(client_uid):
    """
    Event handler for client disconnection.

    Removes the disconnected client from the connected_devices list based on
    the provided client_uid and prints a log message indicating client disconnection.

    Args:
        client_uid (str): The unique identifier of the client.

    Returns:
        None

    Raises:
        None
    """
    device_to_remove = None

    for device in connected_devices:
        if device.get("client_id") == client_uid:
            device_to_remove = device
            break

    if device_to_remove:
        connected_devices.remove(device_to_remove)
        print(f"Client {client_uid} disconnected")


@app.route("/")
def index():
    """
    Renders the index.html template.

    Returns:
        The rendered index.html template.

    Raises:
        None
    """
    return render_template("index.html")


@app.route("/file-names", methods=["GET"])
def get_file_names():
    """
    Retrieves the list of file names from the cloud storage.

    Returns:
        A dictionary containing the list of file names under the key "fileNames".

    Raises:
        None
    """
    return {"fileNames": load_cloud_images()}


@app.route("/collect-frames", methods=["POST", "OPTIONS"])
def handle_collect_frames():
    """
    Request handler for collecting frames.

    Collects frames sent by clients and performs frame processing,
     including landmark extraction and detection.

    Returns a response with appropriate status codes and data.

    Args:
        None

    Returns:
        response (flask.Response): The HTTP response.

    Raises:
        None
    """
    global landmark_lists
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
        except (IOError, OSError) as error:
            print("Error: Failed to open the image:", str(error))

        frameRGB = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            landmark = make_landmark_timestep(results)

            if uid_data not in landmark_lists:
                landmark_lists[uid_data] = []
            landmark_lists[uid_data].append(landmark)
            print("Landmark list length:", len(landmark_lists))

            if len(landmark_lists[uid_data]) == 20:
                thread = threading.Thread(
                    target=detect,
                    args=(
                        model,
                        uid_data,
                        landmark_lists[uid_data],
                    ),
                )
                thread.start()
                landmark_lists[uid_data] = []
                print("Thread started for client:", uid_data)

    except Exception as error:
        print("Error handlingcollect-frame request:", str(error))
        socketio.emit("detection_result", {"status": "0"})

    return jsonify({}), 200


@app.route("/connected-devices", methods=["GET"])
def get_connected_devices():
    """
    Retrieves the list of connected devices.

    Returns the list of connected devices along with
    their client IDs, crime status, and client emails.

    Args:
        None

    Returns:
        response (flask.Response): The HTTP response
        containing the list of connected devices.

    Raises:
        None
    """
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


@app.route("/add-email", methods=["POST"])
def add_email():
    """
    Adds the client email to a connected device.

    Retrieves the client email and client ID from the request
    data and associates the email with the corresponding
    connected device. If the device is not found in the
    connected_devices list, a new entry is created.
    Additionally, a welcome email is sent to the client
    email address. Finally, it returns an empty response with a 200 status code.

    Args:
        None

    Returns:
        response (flask.Response): The empty HTTP response.

    Raises:
        None
    """
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


@app.route("/remove-email", methods=["POST"])
def remove_email():
    """
    Removes the client email from a connected device.

    Retrieves the client email and client ID from the
    request data and searches for a matching device in the
    connected_devices list. If a match is found, the
    client email is removed from the device. If no match is found,
    it returns a JSON response with an error
    message and a 404 status code.

    Args:
        None

    Returns:
        response (flask.Response): The JSON response
        indicating the success or failure of the operation.

    Raises:
        None
    """
    email = request.json["email"]
    uid = request.json["uid"]
    for device in connected_devices:
        if device["client_id"] == uid and device["client_email"] == email:
            device["client_email"] = ""
            return jsonify({}), 200
    return jsonify({"error": "Email not found for the specified client ID."}), 404


@app.before_request
def before_request():
    """
    Middleware function executed before each request.

    Checks if the request method is POST and the request
    path is '/detect-frame'. If the conditions are met, it sets
    the 'CONTENT_TYPE' of the request environment to 'application/json'.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    if request.method == "POST" and request.path == "/detect-frame":
        request.environ["CONTENT_TYPE"] = "application/json"


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
