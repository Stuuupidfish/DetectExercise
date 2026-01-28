GET POINTS WHEN YOU SQUAT OR DO PUSHUPS YAYY

please download the file labeled "Pose landmarker (Full)" from [here](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task) and then put it in this folder for it to work

make sure when your are running this program your full body or most of your body is in view and your background doesnt have a lot of clutter so it can pick up on you as the subject and not mistake any background as the subject.

I tested this for a side view so if you put your camera to the side of your body it should pick it up. Technically it should work at any angle since the points are mapped in 3D coordinates but I suggest setting up your camera in a way that obvious to detect.

for full transparency the exercise detection is my own work. The pose landmark setup was half copy pasted from the documentation and half vibecoded lol
