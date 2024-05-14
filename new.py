import cv2
import numpy as np
cap = cv2.VideoCapture("21.08.mp4")
#create a background object
backgroundobject = cv2.createBackgroundSubtractorMOG2(history=2)
kernel = np.ones((3,3),np.uint8)
kernel2 = None  
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))  # Output file name, codec, fps, and resolution


while True:
    ret, frame = cap.read()
    if not ret:
        break 

    fgmask = backgroundobject.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 20 , 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel2, iterations=6)
    
    #detect the contours 

    countors, _= cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = frame.copy()
    #loop inside the countors and search for biggers 

    for cnt in countors:
        if cv2.contourArea(cnt) > 2000:
            #get the area coordinates
            x , y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy, (x,y),(x+width,y+height),(0,0,255),2)

            #write a text near the object
            cv2.putText(frameCopy ,"object detected", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0),1,cv2.LINE_AA)


    forground = cv2.bitwise_and(frame, frame, mask=fgmask)  

    #combine all together
    stacked = np.hstack((frame,forground,frameCopy))
    cv2.imshow("stacked", cv2.resize(stacked,None, fx=0.4,fy=0.4))


    output.write(frameCopy)
    #cv2.imshow("forground", forground)
    #cv2.imshow("frameCopy", frameCopy)
    # cv2.imshow("fgmask", fgmask)
    #cv2.imshow("img", frame)

    if cv2.waitKey(1) == ord('q'):
        break 

output.release() 
cap.release()
cv2.destroyAllWindows()


