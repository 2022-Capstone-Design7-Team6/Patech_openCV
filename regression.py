import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import sklearn.svm as svm
import math



inputX = [1,2,3,4,5,6,7,8,9,10] 
inputY = [1,2,4,8,16,21,24,26,27,28]
# inputX = [1,2,3,4,5,6] 
# inputY = [1,2,4,8,16,21]

#criteria of harvest
harvestCriteria = 2

#convert X
X = np.array([[he] for he in inputX])
minError = [0,1000000000] #minError maxWeight, minErrorValue, minError 수확날짜?!?
for i in range(max(inputY)+1,70):
    tempError = 0
    #log(-y/y-1) = x 임을 이용!!!(0<y<1 이어야함)  이렇게 하면 y= 1/(1+e^-(ax+b)) 를 예측 가능!  
    maximumWeight = i #이 값에 따라 너무 많이 바뀜.... 그럼 maxumumSize를 inputY 최대값에서부터 하나씩 ..? 늘려가면서 최대무게 측정! 
    #convert Y
    reductY = np.divide(np.array(inputY),maximumWeight) #Later, we have to convert this. e^(logTheY) = input Y no is not......
    #log(-y/(y-1)) = x
    Y = np.log(np.negative(reductY)/(reductY-1))
    model = LinearRegression()
    model.fit(X,Y)
    for i in range(len(inputX)):
        ex = math.exp(model.coef_*inputX[i]+model.intercept_)
        tempError+= abs(ex*maximumWeight/(1+ex)-inputY[i]) #평균절대오차
    # print(maximumWeight,tempError)
    if tempError<minError[1]:
        minError[1]=tempError
        minError[0]=maximumWeight

maximumWeight = minError[0]
reductY = np.divide(np.array(inputY),maximumWeight)
Y = np.log(np.negative(reductY)/(reductY-1))
model = LinearRegression()
model.fit(X,Y)

print(minError)    

if (minError[0]-harvestCriteria) < max(inputY):
    print("추천 수확 시기 : ",max(inputX))
else:
  #y = minError[0]-5 ,  log(-y/(y-1)) = x
   print(model.coef_,model.intercept_)
   tempY = (minError[0]-harvestCriteria)/minError[0]
   tempX = (math.log(-tempY/(tempY-1))-model.intercept_)//model.coef_
   print("추천 수확 시기 : ",int(tempX[0]))


xs = np.arange(0,50,1)
ex =np.exp(model.coef_*xs+model.intercept_)
ys = ex*maximumWeight/(1+ex)
plt.scatter(inputX,inputY,  alpha=0.3)
plt.plot(xs,ys,'r-',lw=3)
plt.show()