from flask import Flask, jsonify, render_template, Response, request
import cv2
import numpy as np
import base64

app = Flask(__name__)


def detect_movement(image_data):
    # Convert the base64 image data to a NumPy array
    nparr = np.frombuffer(base64.b64decode(image_data.split(",")[1]), np.uint8)

    # Decode the image array using OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Set the first frame as the background frame
    if "background" not in detect_movement.__dict__:
        detect_movement.background = image

    # Calculate the absolute difference between the current frame and the background frame
    diff = cv2.absdiff(image, detect_movement.background)

    # Apply a threshold to the difference image
    threshold = 30
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of non-zero pixels in the thresholded image
    nonzero_percentage = np.count_nonzero(thresholded) / thresholded.size

    # Set a threshold for movement detection
    movement_threshold = 0.20

    # Return True if the percentage of non-zero pixels exceeds the movement threshold
    return nonzero_percentage > movement_threshold


def detect_crime(image_data):
    # Perform crime detection logic here
    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-image", methods=["POST"])
def process_image():
    image_data = request.json["image"]

    # Perform movement detection processing on the received image data
    has_movement = detect_movement(image_data)

    has_crime = detect_crime(image_data)

    return jsonify({"has_movement": has_movement, "has_crime": has_crime})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
