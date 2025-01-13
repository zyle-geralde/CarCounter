from ultralytics import YOLO
import cvzone
import cv2
import math

cam = cv2.VideoCapture("./Video/vehiclevideo.mp4")
cam.set(3,960)
cam.set(4,540)

model = YOLO("./models/yolov8n.pt")

predictionNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("./Video/mask.png")

while True:
    success,image = cam.read()
    imgRegion = cv2.bitwise_and(image,mask)
    result = model(imgRegion,stream=True)

    for r in result:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            confidence = (math.ceil(box.conf[0]*100))/100
            class_name = predictionNames[int(box.cls[0])]

            if class_name== "car" and confidence>0.30:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cvzone.putTextRect(image,f"{class_name} {confidence}",(x1,max(33,y1)),scale = 0.8,thickness=1)
    cv2.imshow("Image",image)
    cv2.waitKey(10)

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:  # Window closed
        break
