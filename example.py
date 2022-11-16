import sys
import paCV
import cv2
import numpy
import datetime

#-----EXAMPLE USER DATA-----#
#example user database
class user:
    def __init__(self, ratio, potTopCentimeter,heightList):
        self.ratio =ratio
        self.potTopCentimeter= potTopCentimeter
        self.heightList= heightList
client1 = user(0.35,20,[[datetime.date(2022,11,1),0],[datetime.date(2022,11,15),20],[datetime.date(2022,11,30),30]])
#example img data
img = cv2.imread("C:/Users/minby/Desktop/codes/capstone/before/dd1.jpg")
img2 = cv2.imread("C:/Users/minby/Desktop/codes/capstone/before/p152.jpg")



#-----GUIDE CODE-----#
#def convert2NdArray(img=server image type)
#img = paCV.convert2NdArray(img)
#print('Image type is converted to ',type(img))

#def paImg2AHW(img,paType, ratio,potTopCentimeter)
output = paCV.paImg2AHW(img,0,client1.ratio,client1.potTopCentimeter)
print('Output of paImg2AHW is List : ',output[0],output[1],output[2],'(unit is cm^2, cm, g)')

#def paHarvest(before_img,after_img,ratio, potTopCentimeter)
#output = paCV.paHarvest(img,img2,0,client1.ratio,client1.potTopCentimeter)
#print('Output of paHarvest is List : ',output,'(unit is cm^2, cm, g)')

#def harvPredict(heightList)
#output = paCV.harvPredict(client1.heightList)
#print('Output of harvPredict : ',output)

