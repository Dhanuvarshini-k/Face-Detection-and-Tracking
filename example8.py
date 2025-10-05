import cv2
alg="haarcascade_frontalface_default.xml"
haarcascade=cv2.CascadeClassifier(alg)#Loading algorithm(for every type of loading there will be algorithm)
cam=cv2.VideoCapture(0)#Camera ID Initialization
while True:
    _,img=cam.read()#Reading the frame camera
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#Converting to gray scale image
    face=haarcascade.detectMultiScale(grayImg,1.3,4)#scaling factor and minimal facing pattern(passing them)
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()

