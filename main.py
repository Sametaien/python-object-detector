import cv2


tespit = 0.45 # tespit area

cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

link = "http://192.168.1.103:8080/video"
cap.open(link)


itemIsimleri= []
itemDosyasi = 'coco.names'
with open(itemDosyasi,'rt') as f:
    itemIsimleri = f.read().rstrip('\n').split('\n')

ayarDosyasi = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
ayarDosyasi2 = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(ayarDosyasi2,ayarDosyasi)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=tespit)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,itemIsimleri[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Item Detector",img)
    cv2.waitKey(1)