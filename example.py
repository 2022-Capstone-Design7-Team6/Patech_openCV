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
client1 = user(0.1,20, #datetimeDate, 무게로 ! 수확날짜 중간에 없음! 
               [[datetime.date(2022,10,10),0],
                [datetime.date(2022,10,11),1],
                [datetime.date(2022,10,12),2],
                [datetime.date(2022,10,14),6],
                [datetime.date(2022,10,16),10],
                [datetime.date(2022,10,18),14],
                [datetime.date(2022,10,21),19]])
#example img data
img = cv2.imread("C:/Users/minby/Desktop/codes/capstone/before/real1.jpg")
img2 = cv2.imread("C:/Users/minby/Desktop/codes/capstone/before/p152.jpg")



#-----GUIDE CODE-----#
#def convert2NdArray(img=server image type)
#img = paCV.convert2NdArray(img)
#print('Image type is converted to ',type(img))

#def paImg2AHW(img,paType, ratio,potTopCentimeter)
# output = paCV.paImg2AHW(img,0,client1.ratio,client1.potTopCentimeter)
# print('Output of paImg2AHW is List : ',output[0],output[1],output[2],'(unit is cm^2, cm, g)')

#def paHarvest(before_img,after_img,ratio, potTopCentimeter)
#output = paCV.paHarvest(img,img2,0,client1.ratio,client1.potTopCentimeter)
#print('Output of paHarvest is List : ',output,'(unit is cm^2, cm, g)')

#def harvPredict(heightList)
output = paCV.harvPredict(client1.heightList)
print('Output of harvPredict : ',output)
