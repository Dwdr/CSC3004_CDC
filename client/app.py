import base64
import datetime
import os
import subprocess

from dotenv import load_dotenv

from keras.models import load_model
from keras.optimizers import SGD

import cv2
import numpy as np
import boto3

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

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Save video to AWS S3
@app.route("/save-video", methods=["POST"])
def save_video():
    video = request.files["video"]

    # Configure AWS access keys and region
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = "ap-southeast-1"

    # Specify the S3 bucket name and target path
    bucket_name = "cdcvideobucket"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recorded_video_{timestamp}.webm"
    key = f"videos/{filename}"

    # Upload the video file to Amazon S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name,
    )
    s3.upload_fileobj(video, bucket_name, key)

    # Get the S3 URL for the uploaded file
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"

    # Perform additional processing with the S3 URL or return it as a response
    return jsonify({"s3_url": s3_url})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)