import cv2
import numpy as np
import time,math
import handtrackdetect as htd
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

volbar=400
wcam,hcam=640,480
ptm=0
ctm=0
cap=cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
detect=htd.handDetector(detectionCon=0.65)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=(volume.GetVolumeRange())
print(volRange)
minVol=volRange[0]
maxVol=volRange[1]

while(True):
    success,img=cap.read()
    img=detect.findHands(img)
    lms=detect.findPosition(img,draw=False)
    if(len(lms)>0):
        x1,y1=lms[4][1],lms[4][2]
        x2,y2=lms[8][1],lms[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(x1,y1),10,(255,0,0),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,0),cv2.FILLED)
        cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
        length=(math.hypot(x2-x1,y2-y1))
        # print(length)
        if(length<30):
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
        vol=(np.interp(length,[22,240],[minVol,maxVol]))
        volbar=(np.interp(length,[22,240],[400,150]))
        volume.SetMasterVolumeLevel(vol, None)    
    # print(lms)
    cv2.rectangle(img,(50,150),(85,400),(0,0,255),3)
    cv2.rectangle(img,(50,int(volbar)),(85,400),(0,0,255),cv2.FILLED)
    ctm=time.time()
    fps=1/(ctm-ptm)
    ptm=ctm
    cv2.putText(img,f"FPS:{(int(fps))}",(10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
    cv2.imshow('gesturevolume',img)
    if(cv2.waitKey(1)==27):
        break
cv2.destroyAllWindows()
    