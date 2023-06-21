from flask import Flask, render_template, request, jsonify
# from keras.models import load_model
# from keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

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
    print("Decoding image data...")
    try:
        # Load the pre-trained violence detection model
        sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

        model = load_model("modelnew.h5")
        model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss="categorical_crossentropy", metrics=["accuracy"])

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
    return {"fileNames": load_cloud_videos()}


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
        has_crime = int(detect_crime(image_data))  # Convert boolean to integer
        print({"has_crime": has_crime})
        return jsonify({"has_crime": has_crime})
    except Exception as e:
        print("Error handling detect-frame request:", str(e))
        return jsonify({"error": "Bad Request"}), 400



@app.before_request
def before_request():
    if request.method == "POST" and request.path == "/detect-frame":
        request.environ["CONTENT_TYPE"] = "application/json"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
