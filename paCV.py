import numpy as np
import cv2
import sys
from  PIL  import Image

#카메라 위치 고려하여 길이를 측정 기능?

#상태 : 구현 전(input type 필요)
#기능 : 이미지 파일 타입 변환
#입력 : 서버 상 이미지 type
#출력 : ndarray (dtype = uint8)
def convert2NdArray(img):  #change type to ndarray and dtype is np.uint8  !!!타입을 알아야함
    return img

#상태 : 업그레이드 중
#기능 : 파 넓이 계산
#입력 : image=ndarray , ratio=0~1, potTopCentimeter=cm
#출력 : 넓이(cm^2), 높이(cm)
def paPic(img,ratio,potTopCentimeter):#파사진을 찍었을 때 맨위 위치의 위로 파란색부분을 찾아 넓이계산
    potTopPixel =int(len(img)*ratio)
    #RGB로 특정색을 추출하면 어두운 사진에서 정확도가 떨어짐
    #HSV로 진행. H가 색깔, S가 채도(높으면 선명해짐), V가 명도(낮으면 어두어짐)
    original = img #출력원할떄..
    newImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #명도 high case 
    lower_green = (30, 25, 100)
    upper_green = (90, 255, 255)
    green_mask = cv2.inRange(newImg, lower_green, upper_green)
    #채도 high case
    lower_green = (30, 100, 25)
    upper_green = (90, 255, 255)
    green_mask2 = cv2.inRange(newImg, lower_green, upper_green)
    #여러케이스를 합함
    green_mask+=green_mask2
    #top 아래는 모두 0으로 바꿈
    green_mask[len(img)-potTopPixel:, :]=0
    
    #if you want to see output..
    newImg = cv2.bitwise_and(original, original, mask = green_mask)
    cv2.imwrite('result6.png', newImg)
    
    #calculate area
    countPixel=np.count_nonzero(green_mask)
    countAllPixel = len(img)*len(img[0])
    heightCM = potTopCentimeter/ratio
    widthCM= heightCM*len(img[0])/len(img)
    allArea = heightCM*widthCM
    
    return round(countPixel*allArea/countAllPixel,1)
    
#상태 : 구현완료
#기능 : 두 이미지 파 넓이 차이 계산
#입력 : before_image=ndarray , after_image=ndarray , ratio=0~1, potTopCentimeter=cm
#출력 : 넓이(cm^2)
def paHarv(before_img,after_img,ratio, potTopCentimeter):#수확시, 두 파사진이 동시에 왔을 때 차를 반환 완료
    areaDiff= paPic(after_img,ratio, potTopCentimeter)-paPic(before_img,ratio, potTopCentimeter) 
    if areaDiff<=0 :
        return 'ERROR, pa is grown..'
    else :
        return areaDiff
    
#상태 : 구현 전 평균값 가져와야함
#기능 : 파의 넓이를 무게로 환산
#입력 : pakind=종류(대파=0,쪽파=1,양파=2) area=넓이(cm^2)
#출력 : 무게(g)
def area2weight(pakind,area):
    if pakind ==0:
        pass
    elif pakind ==1:
        pass
    elif pakind==2:
        pass
    else :
        print("ERROR: wrond pa ID")
        return -1
    
pass