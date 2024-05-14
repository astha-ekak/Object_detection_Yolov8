# from ultralytics import YOLO
# import cv2
# import json 
# from datetime import datetime
# # Load the YOLO model
# model = YOLO('yolov5s.pt')

# # Load an image
# image_path = 'images.jpeg'
# image = cv2.imread(image_path)
# results = model(image)  
# x1 = results[0].boxes[0].xyxy[0][0].tolist() 
# y1 = results[0].boxes[0].xyxy[0][1].tolist()
# x2 = results[0].boxes[0].xyxy[0][2].tolist()
# y2 = results[0].boxes[0].xyxy[0][3].tolist() 
# print(results[0].boxes[0].xyxy[0][0]) 

# list1=[x1,y1,x2,y2] 
# timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #confidence = results[0].boxes[0].conf 
# cls_list = results[0].boxes[0].cls.tolist()
# conf_list = results[0].boxes[0].conf.tolist()

# data = {
#     'coordinates': list1,
#     'timestamp': timestamp,
#     'cls': cls_list,
#     'conf': conf_list

#     #'confidence':confidence
# #     'y1': y1,
# #     'x2': x2,
# #     'y2': y2
# } 


# # Dump the dictionary to a JSON file
# with open('output.json', 'w') as json_file:
#     json.dump(data, json_file)

# # # print("JSON file created successfully.")
from ultralytics import YOLO             #importing yolo model from ultralytics to process object detection in video
import cv2                               # importing opencv to capture the video 
import json                              #For storing and exchanging the data 
from datetime import datetime            #importing datetime class from datetime library
import torch                             #for Tensor Computation (similar to NumPy) with strong GPU (Graphical Processing Unit) acceleration support
import numpy as np                       # for n-dimensional array processing and numerical computing.
from loguru import logger  

logger.add('f_second_logging.log',level='INFO',rotation='1 KB') 

# Load the YOLO model
model = YOLO('yolov5su.pt')           

# Open a video file
video_path = 'test1.mp4'
cap = cv2.VideoCapture(video_path) 
frame_n = 0 


data={'objects':[]}                     #creation of dictionary to store json data 
while cap.isOpened():
    ret, frame = cap.read()             #read a frame 
    if not ret:
        break
    results = model(frame)               #This passes the frame through the YOLO model to detect objects.
    logger.info("Code Started")             
    # frame_n+=1 
    for result in results: 
        x1 = result.boxes.xyxy.tolist()  #extract the coordinates of the detected object and then convert it into list
        x3 = result.boxes.conf           #extraction of confidence values of the detected objects .
        a = x3.numpy()                   #conversion of confidence into numpy array 
        b = np.round(a, 2).tolist()       
        values_str = ', '.join(str(x) for x in b)
        x5 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        frame_n+=1

        dict = { 
        'coordinates':x1,
        'confidence':values_str,
        'timestamp':x5,
        'frame':frame_n

        } 
        data['objects'].append(dict)     
        #print(data)
        
        with open('result2.json','w') as file:        # opens a JSON file named 'result2.json' in write mode and writes the data dictionary to it with indentation for better readability.
            json.dump(data,file, indent=4) 
        logger.debug("code end here")
        


        


    

 











        