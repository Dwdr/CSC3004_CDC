import base64
import datetime
import os
import subprocess

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    render_template,
    request,
    send_file,
)
from moviepy.editor import AudioFileClip, VideoFileClip
import boto3
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv() # Load environment variables

# Configure AWS access keys and region
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = "ap-southeast-1"

# Specify the S3 bucket name
s3 = boto3.client(
    "s3", 
    aws_access_key_id=access_key, 
    aws_secret_access_key=secret_key, 
    region_name=region_name
)
bucket_name = "cdcvideobucket"

def detect_movement(image_data):
    nparr = np.frombuffer(base64.b64decode(image_data.split(",")[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if "background" not in detect_movement.__dict__:
        detect_movement.background = image
    diff = cv2.absdiff(image, detect_movement.background)
    threshold = 30
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    nonzero_percentage = np.count_nonzero(thresholded) / thresholded.size
    movement_threshold = 0.20
    return nonzero_percentage > movement_threshold


def detect_crime(image_data):
    # TODO: Implement crime detection
    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-image", methods=["POST"])
def process_image():
    image_data = request.json["image"]
    has_movement = detect_movement(image_data)
    has_crime = detect_crime(image_data)

    return jsonify({"has_movement": has_movement, "has_crime": has_crime})


@app.route("/save-video", methods=["POST"])
def save_video():
    video = request.files["video"]

    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recorded_video_{timestamp}.webm"

    # Upload the video file to Amazon S3 with the target path
    key = f"videos/{filename}"
    s3.upload_fileobj(video, bucket_name, key)
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"

    return jsonify({"s3_url": s3_url})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
