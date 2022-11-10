import sys
import paCV
import cv2
#example user data
class user:
    def __init__(self, ratio, potTopCentimeter):
        self.ratio =ratio
        self.potTopCentimeter= potTopCentimeter
client1 = user(0.45,20)


#type of img is numpy.ndarray
img = cv2.imread("C:/Users/minby/Desktop/pictures/pas/p10_10/G1.jpg")
img = paCV.convert2NdArray(img)


#img_trans = paCV.picTrans(img)


#using paPic
img = cv2.imread("C:/Users/minby/Desktop/codes/capstone/G1.jpg")
print(paCV.paPic(img,client1.ratio,client1.potTopCentimeter))

print(paCV.area2weight(0,20))