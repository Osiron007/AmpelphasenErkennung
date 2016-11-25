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
        colorname = "green"
    elif color == 2:
        dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/yellow"
        colorname = "yellow"
    elif color == 3:
        dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/red"
        colorname = "red"
    elif color == 4:
        dir = "/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/yellowred"
        colorname = "yellowred"

    print(colorname)

    frameCnt = 0

    for filename in os.listdir(dir):
        frameCnt = frameCnt + 1
        pathToFile = str(dir) + "/" + str(filename)
        newPathToFile = str(dir) + "/" + str("tmp%d.jpg" % frameCnt)
        os.rename(pathToFile, newPathToFile)

    frameCnt = 0
    for filename in os.listdir(dir):
        frameCnt = frameCnt + 1
        pathToFile = str(dir) + "/" + str(filename)
        newPathToFile = str(dir) + "/" + str(colorname) + str("%d.jpg" % frameCnt)
        os.rename(pathToFile, newPathToFile)
    i = i + 1
