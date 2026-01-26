#https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#live-stream
#https://github.com/opencv/opencv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np

import os
should_quit = False
latest_annotated_frame = None

model_path = 'pose_landmarker_full.task'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [(11,12),(11,23),(12,24),(23,24), #torso
                    (12,14),(14,16),(16,22),(16,20),(18,20),(16,18), #left arm
                    (11,13),(13,15),(15,21),(15,19),(15,17),(17,19), #right arm
                    (24,26),(26,28),(28,30),(30,32),(28,32), #left leg
                    (23,25),(25,27),(27,29),(27,31),(29,31)] #right leg

pushupCount = 0
lShoulder = 12
lElbow = 14
lWrist = 16
rShoulder = 11
rElbow = 13
rWrist = 15

def getEdges(result):
    # Calculate the left upper arm vector (shoulder to elbow)
    # lShoulder and lElbow are indices defined above
    edges = []
    if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
        landmarks = result.pose_landmarks[0]

        shoulder = landmarks[lShoulder]
        elbow = landmarks[lElbow]
        wrist = landmarks[lWrist]
        
        lUpperArm = (
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        )
        edges.append(lUpperArm)
        lForearm = (
            elbow.x - wrist.x,
            elbow.y - wrist.y,
            elbow.z - wrist.z
        )
        edges.append(lForearm)
        leftEdge = (
        wrist.x - shoulder.x,
        wrist.y - shoulder.y,
        wrist.z - shoulder.z
        )
        
        shoulder = landmarks[rShoulder]
        elbow = landmarks[rElbow]
        wrist = landmarks[rWrist]
        rUpperArm = (
            elbow.x - shoulder.x,
            elbow.y - shoulder.y,
            elbow.z - shoulder.z
        )
        edges.append(rUpperArm)
        rForearm = (
            elbow.x - wrist.x,
            elbow.y - wrist.y,
            elbow.z - wrist.z
        )
        edges.append(rForearm)
        rightEdge = (
        wrist.x - shoulder.x,
        wrist.y - shoulder.y,
        wrist.z - shoulder.z
        )

        edges.append(leftEdge)
        edges.append(rightEdge)
        return edges
    return None

state = "up"
pushupCount = 0
def detectPushup(result):
    #maybe first check if torso/legs are certain degrees from ground to prevent cheating standing up 
    global state, pushupCount
    angles = getAngles(result)
    leftElbow = angles[0]
    rightElbow = angles[1]

    
    if leftElbow >= 150 or rightElbow >= 150:
        if state == "down":
            pushupCount += 1
            state = "up"
        elif state != "up":
            state = "up"
    #down state
    elif leftElbow < 90 and rightElbow < 90:
        if state == "up":
            state = "down"
    
    print(state)

def getAngles(result: PoseLandmarkerResult):
    #check if elbow angle > 160
    edges = getEdges(result)
    
    lUpperArm = edges[0]
    lForearm = edges[1]
    rUpperArm = edges[2]
    rForearm = edges[3]
    
    #build an edge connecting shoulder to wrist to make a triangle
    #law of cosine
    landmarks = result.pose_landmarks[0]
    
    leftEdge = edges[4]
    rightEdge = edges[5]

    #left elbow angle
    leftEdgeMag = np.linalg.norm(leftEdge)
    lUpperMag = np.linalg.norm(lUpperArm)
    lForearmMag = np.linalg.norm(lForearm)
    leftAngle = np.arccos((leftEdgeMag**2 - lUpperMag**2 - lForearmMag**2)/(-2*lUpperMag*lForearmMag))
    
    #right elbow angle
    rightEdgeMag = np.linalg.norm(rightEdge)
    rUpperMag = np.linalg.norm(rUpperArm)
    rForearmMag = np.linalg.norm(rForearm)
    rightAngle = np.arccos((rightEdgeMag**2 - rUpperMag**2 - rForearmMag**2)/(-2*rUpperMag*rForearmMag))
    
    elbowAngles = [np.degrees(leftAngle),np.degrees(rightAngle)]
    return elbowAngles

# def pushupDownstate(image, result):
#     # check if dot prod is pos (angle < 90)
#     edges = getEdges(image, result)
    
#     lUpperArm = edges[0]
#     lForearm = edges[1]
#     rUpperArm = edges[2]
#     rForearm = edges[3]

#     #check left arm
#     return np.dot(lUpperArm, lForearm) >=0 and np.dot(rUpperArm, rForearm) >= 0


def draw_landmarks_on_image(image: np.ndarray, result: PoseLandmarkerResult):
    if not result.pose_landmarks:
        return image
    
    for landmarks in result.pose_landmarks:
        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            h, w, _ = image.shape
            # Convert normalized coordinates to pixel
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        # Draw keypoints, skipping face landmarks (indices 0-10)
        for idx, lm in enumerate(landmarks):
            if 0 <= idx <= 10:
                continue  # Skip face keypoints
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
    
    return image

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # global should_quit
    global latest_annotated_frame
    if result is None or not result.pose_landmarks:
        return
    latest_annotated_frame = draw_landmarks_on_image(output_image.numpy_view().copy(), result)
   
    detectPushup(result)
    print(pushupCount)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

#with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...

frame_count = 0
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0) # 0 for default webcam
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally if needed
        frame_count += 1
        if frame_count % 2 == 0:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            landmarker.detect_async(mp_image, frame_timestamp_ms)
        # Show the latest annotated frame if available
        if latest_annotated_frame is not None:
            cv2.imshow("Video", latest_annotated_frame)
        else:
            cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        