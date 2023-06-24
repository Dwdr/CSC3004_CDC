import cv2
import mediapipe as mp
import pandas as pd

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe pose detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Initialize variables
lm_list = []
label = "neutral"
no_of_frames = 600
i=0

# Function to convert pose landmarks to a flattened list
def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# Function to draw landmarks on the frame
def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

# Capture frames and process pose detection until reaching the desired number of frames
while len(lm_list)<=no_of_frames:
    ret, frame = cap.read()
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            frame = draw_landmark_on_image(mpDraw, results, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Convert the landmark list to a pandas DataFrame and save as a CSV file
df = pd.DataFrame(lm_list)
df.to_csv(label+".txt")

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()