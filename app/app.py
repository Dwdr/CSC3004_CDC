import base64
import datetime
import os
import subprocess

from keras.models import load_model
from keras.optimizers import SGD

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

app = Flask(__name__)


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
    # Load the pre-trained violence detection model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model = load_model('modelnew.h5')

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

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


@app.route("/process-image", methods=["POST"])
def process_image():
    image_data = request.json["image"]
    has_movement = detect_movement(image_data)
    has_crime = detect_crime(image_data)

    return jsonify({"has_movement": has_movement, "has_crime": has_crime})


# TODO: Save video to cloud storage
@app.route("/save-video", methods=["POST"])
def save_video():
    video = request.files["video"]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recorded_video_{timestamp}.webm"

    directory = "resources"
    os.makedirs(directory, exist_ok=True)

    video_path = os.path.join(directory, filename)
    video.save(video_path)

    response = make_response(send_file(video_path))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
