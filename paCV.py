import numpy as np
import cv2
import sys
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

#상태 : input type 필요
#기능 : 이미지 파일 타입 변환
#입력 : 서버 상 이미지 type
#출력 : ndarray (dtype = uint8)
def convert2NdArray(img):  #change type to ndarray and dtype is np.uint8  !!!타입을 알아야함
    if type(img) !=np.ndarray :
        print("Type is not numpy.ndarray")
    return img

#상태 : 업그레이드 중 두께 가중치 추가 고려..
#기능 : 파 넓이 계산
#입력 : image=ndarray , pakind=종류(대파=0,쪽파=1,양파=2) ,ratio=0~1, potTopCentimeter=cm
#출력 : [넓이(cm^2), 높이(cm), 무게(g)]
def paImg2AHW(img,paType, ratio,topCentimeter):#파사진을 찍었을 때 맨위 위치의 위로 파란색부분을 찾아 넓이계산
    wantToReturnOutputImg = True
    
    area2weight = [0.35385,0.16667,0.13846]#대파, 쪽파, 양파
    pxH = len(img)
    pxW = len(img[0])
    potTopPixel =int(pxH*ratio)
    #RGB로 특정색을 추출하면 어두운 사진에서 정확도가 떨어짐
    #HSV로 진행. H가 색깔, S가 채도(높으면 선명해짐), V가 명도(낮으면 어두어짐)
    
    #if you want to see output..1
    if wantToReturnOutputImg:
        original = img 
    
    newImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_bound = [
                    [(25, 200, 5),(97, 255, 100)], #S too high case
                    [(20, 80, 24),(90, 255, 255)], #S high case
                    [(30, 40, 20),(90, 80, 255)], #S mid case
                    [(90, 45, 130),(95, 70, 255)],#S mid and H high case
                    [(45, 20, 50),(89, 50, 255)] #S low case
                    ]
    
    green_mask = cv2.inRange(newImg, green_bound[0][0], green_bound[0][1])
    for i in range(1,len(green_bound)):
        green_mask+=cv2.inRange(newImg, green_bound[i][0], green_bound[i][1])

    #top 아래는 모두 0으로 바꿈
    green_mask[pxH-potTopPixel:, :]=0
    
    #if you want to see output..2
    if wantToReturnOutputImg:
        newImg = cv2.bitwise_and(original, original, mask = green_mask)
    cv2.imwrite("C:/Users/minby/Desktop/codes/capstone/after/result1.png",newImg)
    #cv2.imshow('result',newImg)
    #cv2.waitKey(0)


    
    #calculate area ,height, weight
    countGreenPixel=0
    heightRow = 0
    for row in range(pxH):
        temp  =np.count_nonzero(green_mask[row])
        if heightRow==0 and temp >=3: #1 could be not accurate
            heightRow = row
        countGreenPixel+=temp
    countAllPixel = pxH*pxW
    heightCM = topCentimeter/ratio
    widthCM= heightCM*pxW/pxH
    allArea = heightCM*widthCM
    greenArea = round(allArea*countGreenPixel/countAllPixel,1)
    
    heightRows = pxH - heightRow-int(ratio*pxH)
    height = round(heightCM*heightRows/pxH,1)
    
    weight = round(greenArea*area2weight[paType],1)
    
    
    
    #if you want to see output..3
    if wantToReturnOutputImg:
        return [greenArea,height,weight,newImg]
    return [greenArea,height,weight]
    
#상태 : 구현완료
#기능 : 두 이미지 파 넓이 차이 계산
#입력 : before_image=ndarray , after_image=ndarray , ratio=0~1, potTopCentimeter=cm
#출력 : 두 이미지 [넓이(cm^2), 높이(cm), 무게(g)] 의 차
def paHarvest(before_img,after_img,paType,ratio, potTopCentimeter):#수확시, 두 파사진이 동시에 왔을 때 차를 반환 완료
    diff= [round(a - b,1) for a, b in zip(paImg2AHW(before_img,paType, ratio, potTopCentimeter), paImg2AHW(after_img,paType,ratio, potTopCentimeter) )]
    if diff[0]<0 :
        return 'ERROR, pa is grown..'
    else :
        return diff
    
#상태 : 구현완료 최적화 및 오류제어
#기능 : 성장 곡선 예측, 수확시기 예측
#입력 : heightList = [[datetime1,weight1],[datetime2,weight2],[datetime3,weight3]...]
#출력 : 수확 시기, 수확 시 무게
def harvPredict(weightList,paType):
    #최고 높이를 찾음 거기서 harvestCriteria=2? 가 작은 날을 반환.
    #만약 최고 높이와 현재 식물의 높이가 2가 차이가 안난다고 판단되면 수확을 진행 
    #criteria of harvest
    harvestCriteria = 0.1
    
    firstDay = weightList[0][0]
    for w in weightList:
        w[0] = (w[0]-firstDay).days

    inputX = [ w[0] for w in weightList] #datetime
    inputY =[ w[1] for w in weightList] #weight
    for zero in range(len(inputX)):
        if inputX[zero]==0:
            inputX[zero]+=0.001
    for zero in range(len(inputY)):
        if inputY[zero]==0:
            inputY[zero]+=0.001
    
    #convert inputX shape for regression
    X = np.array([[dates] for dates in inputX])
    
    previousError = 1000000000 #previous error value
    appropriateWeight = 0 #appropriate Weight
    
    curMaxWeight = max(inputY)+0.1
    while(curMaxWeight<=70):
        tempError = 0
        #log(-y/y-1) = x 임을 이용!!!(0<y<1 이어야함)  이렇게 하면 y= 1/(1+e^-(ax+b)) 를 예측 가능!  
        #convert Y
        reductY = np.divide(np.array(inputY),curMaxWeight) #Later, we have to convert this. e^(logTheY) = input Y no is not......
        #log(-y/(y-1)) = x
        Y = np.log(np.negative(reductY)/(reductY-1))
        model = LinearRegression()
        model.fit(X,Y)
        for i in range(len(inputX)):
            ex = math.exp(model.coef_*inputX[i]+model.intercept_)
            tempError+= abs(ex*curMaxWeight/(1+ex)-inputY[i]) #평균절대오차
        # print(curMaxWeight,tempError)
        if tempError<previousError:
            previousError=tempError
            appropriateWeight=curMaxWeight
        # plt.scatter(curMaxWeight,previousError,  alpha=0.3)
        else :
            break
        curMaxWeight+=0.1
    # plt.show()
    # print("Maximum weight of this plant will be",appropriateWeight)
    
    reductY = np.divide(np.array(inputY),appropriateWeight)
    Y = np.log(np.negative(reductY)/(reductY-1))
    model = LinearRegression()
    model.fit(X,Y)

    #if you want to see graph
    xs = np.arange(0,50,1)
    ex =np.exp(model.coef_*xs+model.intercept_)
    ys = ex*appropriateWeight/(1+ex)
    plt.scatter(inputX,inputY,  alpha=0.3)
    plt.plot(xs,ys,'r-',lw=3)
    plt.show()
    
    if (appropriateWeight-harvestCriteria) < max(inputY):
        harvest_date = firstDay + datetime.timedelta(days=max(inputX))
    else:
        tempY = (appropriateWeight-harvestCriteria)/appropriateWeight
        tempX = (math.log(-tempY/(tempY-1))-model.intercept_)//model.coef_
        # print(tempX[0])
        # print(inputX,inputY)
        harvest_date =  firstDay + datetime.timedelta(days=int(tempX[0])+1)
        
        
    #default
    if ((harvest_date-firstDay).days<14):
        harvest_date = firstDay+datetime.timedelta(days=14)
        if (paType==0):appropriateWeight = 25 #대파
        elif (paType==1) : appropriateWeight=10 #쪽파
        else : appropriateWeight=10 #양파파
        
    return [harvest_date,round(appropriateWeight,1)]



