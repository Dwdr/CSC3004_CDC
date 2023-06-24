# CSC3004 Crime Detection System

This repository contains the code for a crime detection system. The system is built using Python and utilizes various technologies and services such as Flask, SocketIO, OpenCV, TensorFlow, AWS S3, SMTP, Docker Desktop and Kubernetes

## Installation

Ensure you have AWS S3 for this project to work. To set up the crime detection system, you may use either Docker, Kubernetes or Python:

### Kubernetes Setup

1. Clone the repository:

```
git clone https://github.com/Dwdr/CSC3004_CDC
```

2. Modify the `app-deployment.yaml` with the environment variables with the following variables:

#### Client

- `PORT`: Port number of the client
- `SERVER_PORT`: Kubernetes NodePort for the server deployment

#### Server

- `PORT`: Port number of the server
- `AWS_ACCESS_KEY_ID`: AWS access key ID for accessing S3 bucket
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key for accessing S3 bucket
- `AWS_REGION: AWS region` where the S3 bucket is located
- `AWS_VIDEO_BUCKET_NAME`: Name of the S3 bucket to store videos
- `AWS_VIDEO_BUCKET_PREFIX`: Prefix for video objects in the S3 bucket
- `AWS_IMAGE_BUCKET_NAME`: Name of the S3 bucket to store images
- `AWS_IMAGE_BUCKET_PREFIX`: Prefix for image objects in the S3 bucket
- `AWS_ANALYSIS_BUCKET_NAME`: Name of the S3 bucket to store analysis results
- `AWS_ANALYSIS_BUCKET_PREFIX`: Prefix for analysis result objects in the S3 bucket
- `EMAIL_ADDRESS`: Email address for sending notifications
- `EMAIL_PASSWORD`: Password for the email address
- `SMTP_SERVER`: SMTP server address for sending emails

3. Deploy the application to Kubernetes by applying the YAML files:

```
kubectl apply -f app-development.yaml
```

The client should be accessible via the NodePort or LoadBalancer service created for the client application. The server dashboard should be accessible via the NodePort or LoadBalancer service created for the server application.

### Docker Setup

1. Clone the repository:

```
git clone https://github.com/Dwdr/CSC3004_CDC
```

2. Set up the environment variables for the server. Run the following command:

```
cd server
cp .env.template .env
```

3. Modify the environment variables within the `.env` file with the following variables:

- `PORT`: Port number for the server
- `AWS_ACCESS_KEY_ID`: AWS access key ID for accessing S3 bucket
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key for accessing S3 bucket
- `AWS_REGION: AWS region` where the S3 bucket is located
- `AWS_VIDEO_BUCKET_NAME`: Name of the S3 bucket to store videos
- `AWS_VIDEO_BUCKET_PREFIX`: Prefix for video objects in the S3 bucket
- `AWS_IMAGE_BUCKET_NAME`: Name of the S3 bucket to store images
- `AWS_IMAGE_BUCKET_PREFIX`: Prefix for image objects in the S3 bucket
- `AWS_ANALYSIS_BUCKET_NAME`: Name of the S3 bucket to store analysis results
- `AWS_ANALYSIS_BUCKET_PREFIX`: Prefix for analysis result objects in the S3 bucket
- `EMAIL_ADDRESS`: Email address for sending notifications
- `EMAIL_PASSWORD`: Password for the email address
- `SMTP_SERVER`: SMTP server address for sending emails

4. Set up the environment variables for the server. Run the following command:

```
cd client
cp .env.template .env
```

5. Modify the environment variables within the `.env` file with the following variables:

- `PORT`: Port number for the server
- `SERVER_PORT`: Port number for the server

6. Ensure that Docker is running and run the following command: `docker compose up`

Client should now be accessible via `127.0.0.1:8000`. Server dashboard should now be accessible via `127.0.0.1:8001`

### Python Setup

1. Clone the repository:

```
git clone https://github.com/Dwdr/CSC3004_CDC
```

2. Set up the environment variables for the server. Run the following command:

```
cd server
cp .env.template .env
```

3. Modify the environment variables within the `.env` file with the following variables:

- `AWS_ACCESS_KEY_ID`: AWS access key ID for accessing S3 bucket
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key for accessing S3 bucket
- `AWS_REGION: AWS region` where the S3 bucket is located
- `AWS_VIDEO_BUCKET_NAME`: Name of the S3 bucket to store videos
- `AWS_VIDEO_BUCKET_PREFIX`: Prefix for video objects in the S3 bucket
- `AWS_IMAGE_BUCKET_NAME`: Name of the S3 bucket to store images
- `AWS_IMAGE_BUCKET_PREFIX`: Prefix for image objects in the S3 bucket
- `AWS_ANALYSIS_BUCKET_NAME`: Name of the S3 bucket to store analysis results
- `AWS_ANALYSIS_BUCKET_PREFIX`: Prefix for analysis result objects in the S3 bucket
- `EMAIL_ADDRESS`: Email address for sending notifications
- `EMAIL_PASSWORD`: Password for the email address
- `SMTP_SERVER`: SMTP server address for sending emails

4. Set up the environment variables for the server. Run the following command:

```
cd client
cp .env.template .env
```

5. Modify the environment variables within the `.env` file with the following variables:

- `PORT`: Port number for the server
- `SERVER_PORT`: Port number for the server

6. Run the client by executing the following command:

```
cd client
python app.py
```

7. Run the server by executing the following command:

```
cd server
python app.py
```

Client should now be accessible via `127.0.0.1:8000`. Server dashboard should now be accessible via `127.0.0.1:8001`

## Usage

The crime detection system can be accessed through a web application. Client web application, which is a browser camera, should now be accessible via `127.0.0.1:8000`. Server dashboard should now be accessible via `127.0.0.1:8001`

- `127.0.0.1:8001/`: The home page of the web application, where users can view the crime - detection system and its functionalities.
- `127.0.0.1:8001/file-names` (GET): Returns a JSON object containing the names of all the images - stored in the S3 bucket.
- `127.0.0.1:8001/detect-frame` (POST): Accepts a JSON object containing the client's unique ID(uid) and image data (image). This endpoint processes the image data and detects crime in a separate thread.
- `127.0.0.1:8001/connected-devices` (GET): Returns a JSON
- `127.0.0.1:8001/add-email` (POST): Add a client's email address to receive crime detection notifications. The request should contain the client's unique ID and the email address.
- `127.0.0.1:8001/remove-email` (POST): Remove a client's email address from receiving crime detection notifications. The request should contain the client's unique ID and the email address.

## Model Training

- Running the train-model folder's make_data captures video frames, detects and tracks poses in each frame using Mediapipe, extracts pose landmarks, and saves them as a CSV file for further analysis.
- The train_lstm.py (Long Short-Term Memory) loads pose data from the dataset, creates and trains an LSTM model and saves the trained model for future used to predict if a punch happened or it is neutral
- This lstm-model that is saved is then used in our server's flask application, app.py to detect for violence

## Functionality

- The server uses a pre-trained violence detection model to analyze images and - detect crime.
- When a client sends an image for detection, the server decodes the image data, performs necessary preprocessing, and predicts whether the image contains - violence.
- If violence is detected, the server saves the image to an Amazon S3 bucket under the appropriate category (crime or no_crime).
- The server maintains a list of connected devices and their crime detection - status.
- When a client connects or disconnects from the server, the server updates the list of connected devices accordingly.
- Clients can provide their email addresses to receive crime detection - notifications.
- If violence is detected in an image, the server sends an email notification to the corresponding client.

## Notes

- This code assumes that you have set up an Amazon S3 bucket and configured the necessary access keys and permissions.
- The code uses Flask and SocketIO for the server implementation. You can adapt it to your specific needs or integrate it into your existing server setup.
- Modify the SMTP server configuration according to your email provider.
- Customize the email content and subject in the send_email and send_welcome_email functions according to your requirements.

Feel free to explore and extend the functionality of this code to suit your project's needs!
