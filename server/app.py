from flask import Flask, render_template, request
from dotenv import load_dotenv
from keras.models import load_model
from keras.optimizers import SGD

import base64
import boto3
import cv2
import numpy as np
import os

app = Flask(__name__)


def load_cloud_videos():
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
    bucket_name = "cdcvideobucket"
    prefix = "videos/"

    # List all objects in the S3 bucket with the specified prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" in response:
        # Extract the file names from the response
        file_names = [
            obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".webm")
        ]
        return file_names
    else:
        return []


def detect_crime(image_data):
    # Load the pre-trained violence detection model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model = load_model("modelnew.h5")

    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    # Convert the image data to a NumPy array
    image = np.frombuffer(base64.b64decode(image_data.split(",")[1]), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Predict whether the image contains violence
    prediction = model.predict(image)

    # Return True if the image contains violence, False otherwise
    return prediction[0] > 0.5


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/file-names", methods=["GET"])
def file_names():
    return {"fileNames": load_cloud_videos()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
