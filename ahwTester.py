import sys
import os 
import paCV
import cv2

dir_path = "C:/Users/minby/Desktop/codes/capstone/before/"
ratio = 0.4
topCentimeter = 20


for (root, directories, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        print(file)
        img = cv2.imread(file_path)
        output = paCV.paImg2AHW(img,0,ratio,topCentimeter)
        print('Output of paImg2AHW is List : ',output[0],output[1],output[2],'(unit is cm^2, cm, g)')
        cv2.imwrite("C:/Users/minby/Desktop/codes/capstone/after/"+file,output[3])

