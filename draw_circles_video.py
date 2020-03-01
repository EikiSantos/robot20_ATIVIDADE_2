#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Matheus Dib, Fabio de Miranda"


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import auxiliar as aux
import math

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')
a = 0
b = 0
degree = 0
x_2 = 0
y_2 = 0
x_1 = 0
y_1 = 0
distancia = 0
hipo = 0
coef = 0
coef_y = 0
angulo = 0
# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1
rosa = "#ff004d"
azul = "#0173fe"

print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)


    circles = []
    

    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    #bordas_hsv = cv2.cvtColor(bordas, cv2.COLOR_BGR2HSV)
    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=80,minRadius=5,maxRadius=40)
        # Our operations on the frame come here
    rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rosa1, rosa2 = aux.ranges(rosa)
    azul1, azul2 = aux.ranges(azul)
    maskrosa = cv2.inRange(hsv, rosa1, rosa2)
    
    
    maskazul = cv2.inRange(hsv, azul1, azul2)
    masktotal = maskazul + maskrosa
    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            coef = 0
            # draw the outer circle
            cv2.circle(masktotal,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(masktotal,(i[0],i[1]),2,(0,0,255),3)
            print("TOTAL:", i)
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(maskazul,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(maskazul,(i[0],i[1]),2,(0,0,255),3)
            cv2.circle(maskrosa,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(maskrosa,(i[0],i[1]),2,(0,0,255),3)
            
            coef = int(i[0]) - a
            coef_y = int(i[1]) - b
            
            a = int(i[0])
            b = int(i[1])
            print("COEF:", coef)
            
            modulo_coef = (coef**2)**0.5
            modulo_coef_y = ((coef_y**2)**0.5)
            if modulo_coef < 10:
                x_1 = i[0]
                y_1 = i[1]
            else:
                x_2 = i[0]
                y_2 = i[1]
            delta_y = float(y_1) - float(y_2)
            delta_x = float(x_1) - float(x_2)
            print("1:", x_1, y_1, "2:", x_2, y_2)
            foco = (30*275)/14
            hipo = ( delta_x**2 + delta_y**2  )**0.5
            print("TYPE:", type(hipo))
            if hipo!= 0:
                if modulo_coef > 10 or modulo_coef_y > 10:
                    distancia = foco*14/hipo
                    angulo = math.acos(((delta_x**2)**0.5)/hipo)
                    degree = math.degrees(angulo)
                    print("TAMANHO DA LINHA:", hipo)
                    print("DISTANCIA:", distancia)
            print("ANGULO:", degree)
            print("------------------------------------------------------------")
            cv2.line(masktotal,(x_1,y_1),(x_2,y_2),(255,0,0),5)
    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #cv2.line(bordas_color,(0,0),(511,511),(255,0,0),5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bordas_color,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    cv2.putText(masktotal,'Distancia: {0} cm'.format(int(distancia)) + "    " + "Angulo:" + "%.2f" % degree + " graus",(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Detector de circulos', masktotal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
