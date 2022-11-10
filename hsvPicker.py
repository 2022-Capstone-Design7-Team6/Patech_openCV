import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

#원하는 픽셀의 HSV값을 확인

img = cv2.imread("C:/Users/minby/Desktop/codes/capstone/M1.jpg")

newImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(newImg)
plt.show()
s_val = []
v_val = []
for i in newImg:
    for ii in i:
        if ii[0]>=30 and ii[0]<=90: #find green
            s_val.append(ii[1])
            v_val.append(ii[2])
plt.scatter(s_val,v_val)
plt.show()

print("Do you wanna see any of those pixel?")
a,b = map(int,input().split())

for i in newImg:
    for ii in i:
        if ii[1]>=a-5 and ii[1]<=a+5 and ii[2]<=b+5 and ii[2]>=b-5: #find green
            pass
        else :
            ii[0]=0
            ii[1]=0
            ii[2]=0
            
plt.imshow(newImg)
plt.show()

