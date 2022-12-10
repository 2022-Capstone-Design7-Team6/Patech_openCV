import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ast
import sklearn.svm as svm

#원하는 픽셀의 HSV값을 확인
def color_picker():
    global newImg
    global img
    global lines
    global lines2
    #input file
    img = cv2.imread("C:/Users/minby/Desktop/codes/capstone/before/kakao1.jpg")
    #size division
    divide = 6
    #read green list
    file = open("C:/Users/minby/Desktop/codes/capstone/green.txt",'r')
    lines = file.read()
    lines=lines.strip()
    lines = ast.literal_eval(lines)
    file.close()
    #read not green list
    file2 = open("C:/Users/minby/Desktop/codes/capstone/notGreen.txt",'r')
    lines2 = file2.read()
    lines2=lines2.strip()
    lines2 = ast.literal_eval(lines2)
    file2.close()
    
    
    newImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x = len(img)
    y= len(img[0])
    
    cv2.namedWindow('pick',cv2.WINDOW_NORMAL)
    cv2.imshow("pick", img)
    cv2.resizeWindow("pick",y//divide,x//divide)
    cv2.setMouseCallback("pick",mouse_click)
    while True:
        if cv2.waitKey(0):
            break
    cv2.destroyAllWindows()
    
    #save lists
    file = open("C:/Users/minby/Desktop/codes/capstone/green.txt",'w')
    file.write(str(lines))
    file.close()
    file2 = open("C:/Users/minby/Desktop/codes/capstone/notGreen.txt",'w')
    file2.write(str(lines2))
    file2.close()


def mouse_click(event, x,y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        lines.append(newImg[y][x].tolist())
        print("Added Green Pixel:",newImg[y][x])
        cv2.imshow("pick",img)
    elif event == cv2.EVENT_FLAG_RBUTTON:
        lines2.append(newImg[y][x].tolist())
        print("Added non-Green Pixel:",newImg[y][x])
        cv2.imshow("pick",img)

def color_organize():
    fig = plt.figure()
    ax= fig.gca(projection = '3d')
    
    file = open("C:/Users/minby/Desktop/codes/capstone/green.txt",'r')
    lines = file.read()
    lines=lines.strip()
    lines = ast.literal_eval(lines)
    x=[]
    y=[]
    z=[] 
    for line in lines:
        x.append(line[0])
        y.append(line[1])
        z.append(line[2])
    file.close()
    
    file2 = open("C:/Users/minby/Desktop/codes/capstone/notGreen.txt",'r')
    lines2 = file2.read().strip()
    lines2 = ast.literal_eval(lines2)
    nx=[]
    ny=[]
    nz=[]
    for line2 in lines2:
        nx.append(line2[0])
        ny.append(line2[1])
        nz.append(line2[2])
    file2.close()
    # x= np.array(x)
    # y= np.array(y)
    # z= np.array(z)
    # nX= np.array(nX)
    # nY= np.array(nY)
    # nZ= np.array(nZ)
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')
    ax.scatter(x,y,z,color='green',s=10)
    ax.scatter(nx,ny,nz,color='red',s=10)
    
    file = open("C:/Users/minby/Desktop/codes/capstone/green.txt",'w')
    lines = list(map(tuple,lines))
    lines = set(list(map(tuple,lines)))
    lines = list(lines)
    lines = list(map(list,lines))
    lines.sort(key = lambda x : (x[1],x[0],x[2]))
    file.write(str(lines))
    file.close()
    
    file2 = open("C:/Users/minby/Desktop/codes/capstone/notGreen.txt",'w')
    lines2 = list(map(tuple,lines2))
    lines2 = set(list(map(tuple,lines2)))
    lines2 = list(lines2)
    lines2 = list(map(list,lines2))
    lines2.sort(key = lambda x : (x[1],x[0],x[2]))
    file2.write(str(lines2))
    file2.close()
    lines = set(list(map(tuple,lines)))
    lines2 = set(list(map(tuple,lines2)))
    print("length of green.txt:",len(lines),"\nlength of nonGreen.txt:",len(lines2))
    print("repitition in two txt : ",set(lines)&set(lines2))
    plt.show()

def bound_tester(green_bound):
    file = open("C:/Users/minby/Desktop/codes/capstone/green.txt",'r')
    lines = file.read()
    lines=lines.strip()
    lines = ast.literal_eval(lines)
    file.close()
    
    file2 = open("C:/Users/minby/Desktop/codes/capstone/notGreen.txt",'r')
    lines2 = file2.read()
    lines2=lines2.strip()
    lines2 = ast.literal_eval(lines2)
    file2.close()
    ErrGreen=0
    ErrnGreen=0
    for line in lines:
        found = False
        for g in green_bound:
            if g[0][0]<=line[0] and line[0]<=g[1][0] and g[0][1]<=line[1] and line[1]<=g[1][1] and g[0][2]<=line[2] and line[2]<=g[1][2]:
                found = True
                break
        if found == False:
            print("This green pixel is out of green_bound : ",line)
            ErrGreen+=1
    
    for line2 in lines2:
        found = False
        for g in green_bound:
            if g[0][0]<=line2[0] and line2[0]<=g[1][0] and g[0][1]<=line2[1] and line2[1]<=g[1][1] and g[0][2]<=line2[2] and line2[2]<=g[1][2]:
                found = True
                break
        if found == True:
            print("This non-green pixel is in the green_bound : ",line2)
            ErrnGreen+=1
    print("green.txt: error/all =",ErrGreen,'/',len(lines))
    print("notGreen.txt: error/all =",ErrnGreen,'/',len(lines2))
    
def webCam_boundTester(green_bound):
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            green_mask = cv2.inRange(frame, green_bound[0][0], green_bound[0][1])
            for i in range(1,len(green_bound)):
                green_mask+=cv2.inRange(frame, green_bound[i][0], green_bound[i][1])
            frame = cv2.bitwise_and(frame, frame, mask = green_mask)
            cv2.imshow("press q to exist", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()



#what's your green bound?
# green_bound = [
#                     [(25, 200, 5),(97, 255, 100)], #S too high case
#                     [(20, 80, 24),(90, 255, 255)], #S high case
#                     [(30, 40, 20),(90, 80, 255)], #S mid case
#                     [(90, 45, 130),(95, 70, 255)],#S mid and H high case
#                     [(45, 20, 50),(89, 50, 255)] #S low case
#                     ]

green_bound =[#단일마스크
                    [(30, 80, 80),(70, 255, 255)]
                    ]

mode = 2
if mode==0: # pick HSV
    color_picker()
elif mode==1 : # organize HSV code 
    color_organize()
elif mode ==2: #test green_bound
    bound_tester(green_bound)
elif mode ==3: #use web cam of green masking
    webCam_boundTester(green_bound)
else:
    print("wrong mode")
    pass

