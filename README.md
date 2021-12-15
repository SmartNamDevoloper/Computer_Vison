# Computer_Vison

This repository has different folders each addressing a different approach to action detection.

#### YoloV3_Object_Detection folder

image.py is to detect objects in a static image. It uses opencv and  yolo v3 weights.

video_obj_detection.py is used to detect objects in a video file. This was also implemented using opencv and  yolo v3 weights.

It was implemented by following this video : https://www.youtube.com/watch?v=1LCb1PVqzeY&list=PLkOqw-5MWU0YOZYrSKGf2bcUAkC8k0BPF&index=113 
Download weights and cfg from  https://pjreddie.com/darknet/yolo/
Class names for coco dataset can be downloaded form https://github.com/pjreddie/darknet/blob/master/data/coco.names

Download videos from pexels.com if required for testing


#### Mediapipe folder

The min.py is a basic implementation of the mediapipe and pose_detect.py is modular code of the same with additional function to detect left and right hand.

This link was followed to implement https://www.youtube.com/watch?v=brwgBf6VB0I&ab_channel=Murtaza%27sWorkshop-RoboticsandAI 


#### movenet folder

The multi_person_pose_estimation.py script was implemented with help of this link Tensorflow Multi-Person Pose Estimation with Python // Machine Learning Tutorial - YouTube ![image](https://user-images.githubusercontent.com/57164676/146131760-1f4c04ff-264f-496b-bed2-3c95143af95f.png)

It detects poses for mutiple persons in the video.
