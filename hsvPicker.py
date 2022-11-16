import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ast


#원하는 픽셀의 HSV값을 확인
def color_picker():
    global newImg
    global img
    global lines
    global lines2
    #input file
    img = cv2.imread("C:/Users/minby/Desktop/codes/capstone/after/dd3.jpg")
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
        print(newImg[y][x])
        cv2.imshow("pick",img)
    elif event == cv2.EVENT_FLAG_RBUTTON:
        lines2.append(newImg[y][x].tolist())
        print(newImg[y][x])
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
    lines.sort(key = lambda x : (x[0],x[1],x[2]))
    file.write(str(lines))
    file.close()
    
    file2 = open("C:/Users/minby/Desktop/codes/capstone/notGreen.txt",'w')
    lines2 = list(map(tuple,lines2))
    lines2 = set(list(map(tuple,lines2)))
    lines2 = list(lines2)
    lines2 = list(map(list,lines2))
    lines2.sort(key = lambda x : (x[0],x[1],x[2]))
    file2.write(str(lines2))
    file2.close()
    lines = set(list(map(tuple,lines)))
    lines2 = set(list(map(tuple,lines2)))
    print(len(lines),len(lines2))
    print(set(lines)&set(lines2))
    plt.show()

mode = 0
if mode==0: # pick HSV
    color_picker()
else : # organize HSV code 
    color_organize()

