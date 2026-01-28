#https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#live-stream
#https://github.com/opencv/opencv
#https://docs.opencv.org/4.x/dc/d4d/tutorial_py_table_of_contents_gui.html

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np

import os
import time

showMessage = 0
messageTime = 0

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


lShoulder = 12
lElbow = 14
lWrist = 16
rShoulder = 11
rElbow = 13
rWrist = 15
lHip = 24
rHip = 23
lKnee = 26
rKnee = 25
lAnkle = 28
rAnkle = 27

points = 0

def getEdges(result):
    # Calculate the left upper arm vector (shoulder to elbow)
    # lShoulder and lElbow are indices defined above
    if not result or not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return [None, None]
    landmarks = result.pose_landmarks[0]

    shoulderL = landmarks[lShoulder]
    elbowL = landmarks[lElbow]
    wristL = landmarks[lWrist]
    shoulderR = landmarks[rShoulder]
    elbowR = landmarks[rElbow]
    wristR = landmarks[rWrist]
    hipL = landmarks[lHip]
    hipR = landmarks[rHip]
    kneeL = landmarks[lKnee]
    kneeR = landmarks[rKnee]
    ankleL = landmarks[lAnkle]
    ankleR = landmarks[rAnkle]
    
    lUpperArm = (
        elbowL.x - shoulderL.x,
        elbowL.y - shoulderL.y,
        elbowL.z - shoulderL.z
    )
    lForearm = (
        elbowL.x - wristL.x,
        elbowL.y - wristL.y,
        elbowL.z - wristL.z
    )
    leftEdge = (
    wristL.x - shoulderL.x,
    wristL.y - shoulderL.y,
    wristL.z - shoulderL.z
    )
    
    rUpperArm = (
        elbowR.x - shoulderR.x,
        elbowR.y - shoulderR.y,
        elbowR.z - shoulderR.z
    )
    rForearm = (
        elbowR.x - wristR.x,
        elbowR.y - wristR.y,
        elbowR.z - wristR.z
    )
    rightEdge = (
    wristR.x - shoulderR.x,
    wristR.y - shoulderR.y,
    wristR.z - shoulderR.z
    )

    leftTorsoEdge = (
        hipL.x - shoulderL.x,
        hipL.y - shoulderL.y,
        hipL.z - shoulderL.z
    )
    rightTorsoEdge =  (
        hipR.x - shoulderR.x,
        hipR.y - shoulderR.y,
        hipR.z - shoulderR.z
    )

    leftThigh = (
        hipL.x - kneeL.x,
        hipL.y - kneeL.y,
        hipL.z - kneeL.z
    )
    rightThigh = (
        hipR.x - kneeR.x,
        hipR.y - kneeR.y,
        hipR.z - kneeR.z
    )
    
    leftShin = (
        ankleL.x - kneeL.x,
        ankleL.y - kneeL.y,
        ankleL.z - kneeL.z
    )
    rightShin = (
        ankleR.x - kneeR.x,
        ankleR.y - kneeR.y,
        ankleR.z - kneeR.z
    )
    
    edges = [lUpperArm, lForearm, rUpperArm, 
             rForearm, leftEdge, rightEdge, 
             leftTorsoEdge, rightTorsoEdge, 
             leftThigh, leftShin, 
             rightThigh, rightShin]

    return edges

pState = "up"
pushupCount = 0
def detectPushup(result):    
    global pState, pushupCount, points, showMessage, messageTime
    bodyAngles = getBodyAngle(result)
    angles = getElbowAngles(result)
    leftElbow = angles[0]
    rightElbow = angles[1]
    #print(bodyAngles)
    if (bodyAngles[0] > 50 or bodyAngles[1] > 50):
        print("torso down")

        if leftElbow >= 150 or rightElbow >= 150:
            if pState == "down":
                pushupCount += 1
                points += 2
                showMessage = 1
                messageTime = time.time()
                pState = "up"
            elif pState != "up":
                pState = "up"
        #down state
        elif leftElbow < 110 and rightElbow < 110:
            if pState == "up":
                pState = "down"
        
    #print(pState)

squatCount = 0
sState = "up"
def detectSquat(result):
    global sState, squatCount, points, showMessage, messageTime
    bodyAngles = getBodyAngle(result)
    angles = getKneeAngles(result)
    leftKnee = angles[0]
    rightKnee = angles[1]
    if (bodyAngles[0] < 50 or bodyAngles[1] < 50):
        print("torso up")
        if leftKnee >= 155 or rightKnee >= 155:
            if sState == "down":
                squatCount += 1
                points += 1
                showMessage = 2
                messageTime = time.time()
                sState = "up"
            elif sState != "up":
                sState = "up"
        #down state
        elif leftKnee < 90 and rightKnee < 90:
            if sState == "up":
                sState = "down"
    print(sState)
    return

def getBodyAngle(result):
    #check if torso(shoulder to hip line) x-axis angle is less than 45 degrees
    
    #hi gang if youre reading this youre probably wondering why i use dot product to calculate the angle
    #when i used law of cosines for the elbows and to be honest
    #i was stupid
    #i forgot i could also do this
    j = (0,1,0)
    edges = getEdges(result)
    leftTorsoEdge = edges[6]
    lTorsoMag = np.linalg.norm(leftTorsoEdge)
    leftAngle = np.arccos(np.dot(j,leftTorsoEdge)/lTorsoMag)

    rightTorsoEdge = edges[7]
    rTorsoMag = np.linalg.norm(rightTorsoEdge)
    rightAngle = np.arccos(np.dot(j,rightTorsoEdge)/rTorsoMag)

    torsoAxisAngles = [np.degrees(leftAngle),np.degrees(rightAngle)]
    return torsoAxisAngles

def getKneeAngles(result):
    edges = getEdges(result)
    leftThigh = edges[8]
    leftShin = edges[9]
    rightThigh = edges[10]
    rightShin = edges[11]

    lThighMag = np.linalg.norm(leftThigh)
    lShinMag = np.linalg.norm(leftShin)
    rThighMag = np.linalg.norm(rightThigh)
    rShinMag = np.linalg.norm(rightShin)

    leftKneeAngle = np.arccos(np.dot(leftThigh,leftShin)/(lThighMag*lShinMag))
    rightKneeAngle = np.arccos(np.dot(rightThigh,rightShin)/(rThighMag*rShinMag))

    return [np.degrees(leftKneeAngle),np.degrees(rightKneeAngle)]

def getElbowAngles(result):
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
    #print(pushupCount)
    detectSquat(result)
    print(squatCount)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


#with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...


frame_count = 0
displayImg = None
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
            displayImg = latest_annotated_frame
        else:
            displayImg = frame
        
        cv2.putText(displayImg, "Points: " + str(points) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if showMessage != 0:
            if time.time() - messageTime < 1:
                if showMessage == 1:
                    cv2.putText(displayImg, "Pushup +2" , (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if showMessage == 2:
                    cv2.putText(displayImg, "Squat +1" , (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                showMessage = False

        
        cv2.imshow("Video", displayImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        