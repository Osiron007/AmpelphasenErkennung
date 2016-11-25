
import cv2 as cv

import os

i = 1

while i < 5:
    color = i # 1 =green
              # 2 = yellow
              # 3 = red
              # 4 = yellowred

    dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/green"
    colorname = "green"

    if color == 1:
        dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/green"
        targetdir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder_100px_100px/green"
        targetdirgrey = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Graustufen_100px_100px/green"
        colorname = "green"
    elif color == 2:
        dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/yellow"
        targetdir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder_100px_100px/yellow"
        targetdirgrey = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Graustufen_100px_100px/yellow"
        colorname = "yellow"
    elif color == 3:
        dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/red"
        targetdir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder_100px_100px/red"
        targetdirgrey = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Graustufen_100px_100px/red"
        colorname = "red"
    elif color == 4:
        dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/yellowred"
        targetdir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder_100px_100px/yellowred"
        targetdirgrey = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Graustufen_100px_100px/yellowred"
        colorname = "yellowred"

    print(colorname)

    frameCnt = 0
    for filename in os.listdir(dir):
        frameCnt = frameCnt + 1

        #file locations
        pathToFile = str(dir) + "/" + str(filename)
        newPathToFile = str(targetdir) + "/" + str(colorname) + str("%d.jpg" % frameCnt)
        newPathToGreyFile = str(targetdirgrey) + "/" + str(colorname) + str("%d.jpg" % frameCnt)

        #read image from file
        image = cv.imread(pathToFile)

        #resize image
        resized_image = cv.resize(image,(100,100), interpolation=cv.INTER_CUBIC)

        grey_image = cv.cvtColor(resized_image,cv.COLOR_BGR2GRAY)

        #save resized_image to file
        cv.imwrite(newPathToFile, resized_image)

        # save grey_image to file
        cv.imwrite(newPathToGreyFile, grey_image)

    i = i + 1