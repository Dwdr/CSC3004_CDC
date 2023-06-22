import base64
import io
import binascii
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
load_dotenv() # Load environment variables

# Configure AWS access keys and region
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name = "ap-southeast-1"

# Specify the S3 bucket name and target path
s3 = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region_name,
)
video_bucket_name = "cdcvideobucket"
video_bucket_prefix = "videos/"
image_bucket_name = "cdcimagebucket"
image_bucket_prefix = "images/"
analysis_bucket_name = "cdcanalysisbucket"
analysis_bucket_prefix = "analysis/"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/save-image", methods=["POST"])
def save_image():
    uid = request.json["uid"]
    image_data = request.json["image"]

    # Add padding to the image data if it's missing
    padding_length = len(image_data) % 4
    if padding_length > 0:
        image_data += "=" * (4 - padding_length)
    
    try:
        # Convert the base64 image data to binary
        image_binary = base64.b64decode(image_data)
    except binascii.Error as e:
        return jsonify(
            {
                "success": False, 
                "message": "Invalid base64-encoded image data"
            }
        )
    
    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"image_{uid}_{timestamp}.png"
    key = f"images/{filename}"

    # Upload image to the S3 bucket
    s3.put_object(
        Bucket=image_bucket_name,
        Key=key,
        Body=image_binary,
        ContentType="image/png"
    )

    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
