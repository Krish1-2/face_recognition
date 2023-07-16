import cv2
import os

# assign directory
directory = 'dataset'
# iterate over files in
# that directory

video=cv2.VideoCapture(0)
count=0
print("enter your name:")
name=input()
path=directory+'/'+name
while True:
        count+=1
        ret,frame=video.read()
        cv2.imshow('windowsFrame',frame)
        print(count)
        if not os.path.exists(path):
                os.makedirs(path)
        cv2.imwrite(path+"/image"+str(count)+'.jpg',frame)
        if(count>=500):
                break
video.release()
cv2.destroyAllWindows()

