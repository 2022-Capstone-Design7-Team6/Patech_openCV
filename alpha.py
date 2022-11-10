import numpy as np
import cv2
import sys
from  PIL  import Image

def picTrans(img):#사진을 반투명하게 만듬 완료
    transparency = 127 #min 0. max 255
    b, g, r = cv2.split(img)
    mask=np.full((len(img),len(img[0])),transparency,dtype=np.uint8)
    newImg = cv2.merge([b, g, r, mask], 4)
    # write as png which keeps alpha channel 
    #cv2.imwrite('result.png', newImg)
    return newImg
