import numpy as np
import cv2
import sys
import datetime

#카메라 위치 고려하여 길이를 측정 기능?

#상태 : 구현 전(input type 필요)
#기능 : 이미지 파일 타입 변환
#입력 : 서버 상 이미지 type
#출력 : ndarray (dtype = uint8)
def convert2NdArray(img):  #change type to ndarray and dtype is np.uint8  !!!타입을 알아야함
    if type(img) !=np.ndarray :
        print("Type is not numpy.ndarray")
    return img

#상태 : 업그레이드 중 두께 가중치 추가?
#기능 : 파 넓이 계산
#입력 : image=ndarray , pakind=종류(대파=0,쪽파=1,양파=2) ,ratio=0~1, potTopCentimeter=cm
#출력 : [넓이(cm^2), 높이(cm), 무게(g)]
def paImg2AHW(img,paType, ratio,potTopCentimeter):#파사진을 찍었을 때 맨위 위치의 위로 파란색부분을 찾아 넓이계산
    area2weight = [2,1,1]#대파, 쪽파, 양파
    pxH = len(img)
    pxW = len(img[0])
    potTopPixel =int(pxH*ratio)
    #RGB로 특정색을 추출하면 어두운 사진에서 정확도가 떨어짐
    #HSV로 진행. H가 색깔, S가 채도(높으면 선명해짐), V가 명도(낮으면 어두어짐)
    
    #if you want to see output..1
    original = img 
    
    newImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #명도 high case 
    lower_green = (30, 25, 25)
    upper_green = (90, 120, 255)
    green_mask = cv2.inRange(newImg, lower_green, upper_green)
    #채도 high case
    lower_green = (30, 25, 25)
    upper_green = (90, 255, 120)
    green_mask2 = cv2.inRange(newImg, lower_green, upper_green)
    #색조 high case
    lower_green = (90, 50, 120)
    upper_green = (95, 70, 170)
    green_mask3 = cv2.inRange(newImg, lower_green, upper_green)
    
    #여러케이스를 합함
    green_mask=green_mask+green_mask2+green_mask3
    #top 아래는 모두 0으로 바꿈
    green_mask[pxH-potTopPixel:, :]=0
    
    #if you want to see output..2
    newImg = cv2.bitwise_and(original, original, mask = green_mask)
    cv2.imwrite('result6.png', newImg)
    
    #calculate area ,height, weight
    countGreenPixel=0
    heightRow = 0
    for row in range(pxH):
        temp  =np.count_nonzero(green_mask[row])
        if heightRow==0 and temp >=2: #1 could be not accurate
            heightRow = row
        countGreenPixel+=temp
    countAllPixel = pxH*pxW
    heightCM = potTopCentimeter/ratio
    widthCM= heightCM*pxW/pxH
    allArea = heightCM*widthCM
    greenArea = round(allArea*countGreenPixel/countAllPixel,1)
    
    heightRows = pxH - heightRow-int(ratio*pxH)
    height = round(heightCM*heightRows/pxH,1)
    
    weight = round(greenArea*area2weight[paType],1)
    return [greenArea,height,weight]
    
#상태 : 구현완료
#기능 : 두 이미지 파 넓이 차이 계산
#입력 : before_image=ndarray , after_image=ndarray , ratio=0~1, potTopCentimeter=cm
#출력 : 두 이미지 [넓이(cm^2), 높이(cm), 무게(g)] 의 차
def paHarvest(before_img,after_img,paType,ratio, potTopCentimeter):#수확시, 두 파사진이 동시에 왔을 때 차를 반환 완료
    diff= [round(a - b,1) for a, b in zip(paImg2AHW(after_img,paType, ratio, potTopCentimeter), paImg2AHW(before_img,paType,ratio, potTopCentimeter) )]
    if diff[0]<0 :
        return 'ERROR, pa is grown..'
    else :
        return diff
    
#상태 : 구현전
#기능 : 성장 곡선 예측, 수확시기 예측
#입력 : heightList = [[datetime1,height1],[datetime2,height2],[datetime3,height3]...]
#출력 : 수확 시기...?
def harvPredict(heightList):
    #[[datatime,50]]

    return True