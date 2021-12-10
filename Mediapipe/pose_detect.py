import cv2
import mediapipe as mp
import time
class poseDetector():
    def __init__(self, mode=False, upperBody=False, smooth=True,
                 detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose

        # self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_segmentation=self.smooth,
        #                              min_detection_confidence=self.detectionConf, min_tracking_confidence = self.trackConf)

        self.pose = self.mpPose.Pose()
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            # print(type(self.results.pose_landmarks.landmark))
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w),int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img,(cx,cy), 20, (255,0,0), cv2.FILLED)

        return lmList

def main():
    cap = cv2.VideoCapture('videos/1.mp4')
    pTime = 0
    detector = poseDetector()
    print("main")
    while True:
        success, img = cap.read()
        img  = detector.findPose(img)
        lmlist = detector.findPosition(img,draw=False)
        # print(lmlist)
        if lmlist:
            print(lmlist[15])
            print(lmlist[16])
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0),3)
        if lmlist:
            cv2.circle(img, (lmlist[15][1], lmlist[15][2]), 20, (255, 0, 100), cv2.FILLED)
            cv2.circle(img, (lmlist[16][1], lmlist[16][2]), 20, (255, 0, 255), cv2.FILLED)
        img = cv2.resize(img, (450, 450))

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == "__main__":
    main()