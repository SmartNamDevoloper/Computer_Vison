import cv2
import time
import numpy as np
# load the COCO class names
#save all the names in file o the list classes
classes =[]
with open("./yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# with open('../../input/object_detection_classes_coco.txt', 'r') as f:
#     class_names = f.read().split('\n')

# load the DNN model
net = cv2.dnn.readNet("./yolo/yolov3.weights", "./yolo/yolov3.cfg")

#get the image and its size
# img = cv2.imread("./yolo/bus.jpg")
cap = cv2.VideoCapture("./yolo/video.mp4")
# cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    height, width,_ = img.shape

    #Transform the image into a blob
    blob= cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0), swapRB=True, crop=False)

    #Set the blob as the input to the net
    net.setInput(blob)

    #Get the output layer names and pass to forward pass of opencv
    out_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(out_layer_names)
    # print(out)

    #Now to visualise the result predictions
    boxes = []
    confidences = []
    class_ids = []

    for output in layeroutputs:
        for detection in output:
            #there are 85 alues in detection
            # first 4 show the location of the object
            # the rest give the prob of the 80 classes
            scores = detection[5:]
            class_id = np.argmax(scores) #gives the highest score
            confidence = scores[class_id] #gives the confidence score of the most likey object
            if confidence > 0.5: # confidence threshold can be changed to extract some objects inthe background
                center_x = int(detection[0]*width) #multiplying by width because we normalised the image
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                # yolo predicts the center and width and height
                # to draw the box we need to calculate the 4 corners
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                #append the detected objects box to the list
                boxes.append([x, y, w, h])
                #append the confidence level to list
                confidences.append(float(confidence))
                class_ids.append(class_id)
    #in detection there will be more than one box that is detected to
    #consider only one box we use Non maximal supression(NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #pass all the gathered info to display
    font = cv2.FONT_HERSHEY_PLAIN
    # get a different color array for each of the classes
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    print("Indexes",indexes)
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " +confidence, (x,y-5), font, 2, (255,255,255), 2)
    #had to resize the window to disply properly
    imS = cv2.resize(img, (960, 540))  # Resize image
    cv2.imshow("Image", imS)
    # cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
