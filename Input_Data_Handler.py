import cv2 as cv
import os
import numpy as np
import random as rnd

class InputDataHandler(object):
    images = 0
    __images_green__ = []
    __images_yellow__ = []
    __images_red__ = []
    __images_yellowred__ = []

    __NrOfGreenSignals__ = 0
    __NrOfYellowSignals__ = 0
    __NrOfRedSignals__ = 0
    __NrOfYellowRedSignals__ = 0


    #this handler is initialized when get_images_from_files was executed
    initialized = False

    image_dir = "/home/dlm/AmpelPhasen_Bilder/"

    dir_green_greyscales50x50 = image_dir + str("Graustufen_50px_50px/green")
    dir_yellow_greyscales50x50 = image_dir + str("Graustufen_50px_50px/yellow")
    dir_red_greyscales50x50 = image_dir + str("Graustufen_50px_50px/red")
    dir_yellowred_greyscales50x50 = image_dir + str("Graustufen_50px_50px/yellowred")

    dir_green_classification = image_dir + str("toClassify/green")
    dir_yellow_classification = image_dir + str("toClassify/yellow")
    dir_red_classification = image_dir + str("toClassify/red")
    dir_yellowred_classification = image_dir + str("toClassify/yellowred")

    dir_green_50x50 = image_dir + str("Bilder_50px_50px/green")
    dir_yellow_50x50 = image_dir + str("Bilder_50px_50px/yellow")
    dir_red_50x50 = image_dir + str("Bilder_50px_50px/red")
    dir_yellowred_50x50 = image_dir + str("Bilder_50px_50px/yellowred")

    def get_images_from_files(self):



        green_images_cnt = len(os.listdir(self.dir_green_greyscales50x50))
        yellow_images_cnt = len(os.listdir(self.dir_yellow_greyscales50x50))
        red_images_cnt = len(os.listdir(self.dir_red_greyscales50x50))
        yellowred_images_cnt = len(os.listdir(self.dir_yellowred_greyscales50x50))


        #read green images from files
        fileCnt = 0
        for file in os.listdir(self.dir_green_greyscales50x50):
            pathToFile = self.dir_green_greyscales50x50 + "/" + file
            self.__images_green__.append(cv.imread(pathToFile,cv.IMREAD_GRAYSCALE))
            fileCnt = fileCnt + 1
        print("green filecnt = " + str(fileCnt))
        self.__NrOfGreenSignals__ = fileCnt


        # read yellow images from files
        fileCnt = 0
        for file in os.listdir(self.dir_yellow_greyscales50x50):
            pathToFile = self.dir_yellow_greyscales50x50 + "/" + file
            self.__images_yellow__.append(cv.imread(pathToFile,cv.IMREAD_GRAYSCALE))
            fileCnt = fileCnt + 1
        print("yellow filecnt = " + str(fileCnt))
        self.__NrOfYellowSignals__ = fileCnt

        # read red images from files
        fileCnt = 0
        for file in os.listdir(self.dir_red_greyscales50x50):
            pathToFile = self.dir_red_greyscales50x50 + "/" + file
            self.__images_red__.append(cv.imread(pathToFile,cv.IMREAD_GRAYSCALE))
            fileCnt = fileCnt + 1
        print("red filecnt = " + str(fileCnt))
        self.__NrOfRedSignals__ = fileCnt

        # read yellowred images from files
        fileCnt = 0
        for file in os.listdir(self.dir_yellowred_greyscales50x50):
            pathToFile = self.dir_yellowred_greyscales50x50 + "/" + file
            self.__images_yellowred__.append(cv.imread(pathToFile,cv.IMREAD_GRAYSCALE))
            fileCnt = fileCnt + 1
        print("yellowred filecnt = " + str(fileCnt))
        self.__NrOfYellowRedSignals__ = fileCnt


        initialized = True

    def get_batch(self,batch_size):
        #batchsize has to be batchsize%4 == 0
        batch = []
        labels = []
        errorcode = 0

        rnd.seed()

        greenCnt = 0
        yellowCnt = 0
        redCnt = 0
        yellowRedCnt = 0

        for i in range(batch_size):
            #get rnd class number
            #print(i)
            classNumber = rnd.randint(1,4)

            # check if image should be modified to extend data set
            mirrorImage = rnd.randint(0, 1)
            changeContrast = rnd.randint(0, 1)
            changeBrightness = rnd.randint(0, 1)

            if classNumber == 1:
                #Green
                #print("Green")
                minNr = 0
                maxNr = self.__NrOfGreenSignals__-1
                #print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr,maxNr)
                batch.append(self.__images_green__[imageNr])
                labels.append((1,0,0,0))
                greenCnt = greenCnt +1
            if classNumber == 2:
                #Yellow
                #print("Yellow")
                minNr = 0
                maxNr = self.__NrOfYellowSignals__ - 1
                #print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                batch.append(self.__images_yellow__[imageNr])
                labels.append((0, 1, 0, 0))
                yellowCnt = yellowCnt + 1
            if classNumber == 3:
                #Red
                #print("Red")
                minNr = 0
                maxNr = self.__NrOfRedSignals__ - 1
                #print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                batch.append(self.__images_red__[imageNr])
                labels.append((0, 0, 1, 0))
                redCnt = redCnt + 1
            if classNumber == 4:
                #YellowRed
                #print("YellowRed")
                minNr = 0
                maxNr = self.__NrOfYellowRedSignals__ - 1
                #print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                batch.append(self.__images_yellowred__[imageNr])
                labels.append((0, 0, 0, 1))
                yellowRedCnt = yellowRedCnt + 1

            if mirrorImage == 1:
                batch[i] = cv.flip(batch[i],1)

            #if changeContrast == 1:
            #    batch[i] = cv.flip(batch[i],1)

            #if changeBrightness == 1:
            #    batch[i] = cv.flip(batch[i],1)

            #flattern image matrix
            batch[i] = np.asarray(batch[i])

        #print("Green: " + str(greenCnt) + " Yellow: " + str(yellowCnt) + " Red: " + str(redCnt)+ " YellowRed: " + str(yellowRedCnt))
        return batch , labels, errorcode

    def load_files_and_get_batch(self, batch_size):
        # batchsize has to be batchsize%4 == 0
        batch = []
        labels = []
        errorcode = 0

        green_images_cnt = len(os.listdir(self.dir_green_greyscales50x50))
        yellow_images_cnt = len(os.listdir(self.dir_yellow_greyscales50x50))
        red_images_cnt = len(os.listdir(self.dir_red_greyscales50x50))
        yellowred_images_cnt = len(os.listdir(self.dir_yellowred_greyscales50x50))

        rnd.seed()

        greenCnt = 0
        yellowCnt = 0
        redCnt = 0
        yellowredCnt = 0

        for i in range(batch_size):
            # get rnd class number
            # print(i)
            classNumber = rnd.randint(1, 4)

            # check if image should be modified to extend data set
            mirrorImage = rnd.randint(0, 1)
            changeContrast = rnd.randint(0, 1)
            changeBrightness = rnd.randint(0, 1)

            if classNumber == 1:
                # Green
                # print("Green")
                minNr = 1
                maxNr = green_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "green" + str(imageNr) + ".jpg"
                pathToFile = self.dir_green_greyscales50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE))
                    labels.append((1, 0, 0, 0))
                    greenCnt = greenCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 2:
                # Yellow
                # print("Yellow")
                minNr = 1
                maxNr = yellow_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellow" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellow_greyscales50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE))
                    labels.append((0, 1, 0, 0))
                    yellowCnt = yellowCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 3:
                # Red
                # print("Red")
                minNr = 1
                maxNr = red_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "red" + str(imageNr) + ".jpg"
                pathToFile = self.dir_red_greyscales50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE))
                    labels.append((0, 0, 1, 0))
                    redCnt = redCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 4:
                # yellowred
                # print("yellowred")
                minNr = 1
                maxNr = yellowred_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellowred" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellowred_greyscales50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE))
                    labels.append((0, 0, 0, 1))
                    yellowredCnt = yellowredCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")

            if mirrorImage == 1:
                batch[i] = cv.flip(batch[i], 1)

            # if changeContrast == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # if changeBrightness == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # flattern image matrix
            batch[i] = np.asarray(batch[i])

        # print("Green: " + str(greenCnt) + " Yellow: " + str(yellowCnt) + " Red: " + str(redCnt)+ " YellowRed: " + str(yellowRedCnt))
        return batch, labels, errorcode

    def load_color_files_and_get_batch(self, batch_size):
        # batchsize has to be batchsize%4 == 0
        batch = []
        labels = []
        errorcode = 0


        green_images_cnt = len(os.listdir(self.dir_green_50x50))
        yellow_images_cnt = len(os.listdir(self.dir_yellow_50x50))
        red_images_cnt = len(os.listdir(self.dir_red_50x50))
        yellowred_images_cnt = len(os.listdir(self.dir_yellowred_50x50))

        rnd.seed()

        greenCnt = 0
        yellowCnt = 0
        redCnt = 0
        yellowredCnt = 0

        for i in range(batch_size):
            # get rnd class number
            # print(i)
            classNumber = rnd.randint(1, 4)

            # check if image should be modified to extend data set
            mirrorImage = rnd.randint(0, 1)
            changeContrast = rnd.randint(0, 1)
            changeBrightness = rnd.randint(0, 1)

            if classNumber == 1:
                # Green
                # print("Green")
                minNr = 1
                maxNr = green_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "green" + str(imageNr) + ".jpg"
                pathToFile = self.dir_green_50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile, cv.IMREAD_COLOR))
                    labels.append((1, 0, 0, 0))
                    greenCnt = greenCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 2:
                # Yellow
                # print("Yellow")
                minNr = 1
                maxNr = yellow_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellow" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellow_50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile, cv.IMREAD_COLOR))
                    labels.append((0, 1, 0, 0))
                    yellowCnt = yellowCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 3:
                # Red
                # print("Red")
                minNr = 1
                maxNr = red_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "red" + str(imageNr) + ".jpg"
                pathToFile = self.dir_red_50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile, cv.IMREAD_COLOR))
                    labels.append((0, 0, 1, 0))
                    redCnt = redCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 4:
                # yellowred
                # print("yellowred")
                minNr = 1
                maxNr = yellowred_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellowred" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellowred_50x50 + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(cv.imread(pathToFile))
                    labels.append((0, 0, 0, 1))
                    yellowredCnt = yellowredCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")

            if mirrorImage == 1:
                batch[i] = cv.flip(batch[i], 1)

            # if changeContrast == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # if changeBrightness == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # flattern image matrix
            batch[i] = np.asarray(batch[i])

        # print("Green: " + str(greenCnt) + " Yellow: " + str(yellowCnt) + " Red: " + str(redCnt)+ " YellowRed: " + str(yellowRedCnt))
        return batch, labels, errorcode

    def load_files_and_get_batch_grey_for_classification(self, batch_size):
        # batchsize has to be batchsize%4 == 0
        batch = []
        labels = []
        errorcode = 0

        green_images_cnt = len(os.listdir(self.dir_green_classification))
        yellow_images_cnt = len(os.listdir(self.dir_yellow_classification))
        red_images_cnt = len(os.listdir(self.dir_red_classification))
        yellowred_images_cnt = len(os.listdir(self.dir_yellowred_classification))

        rnd.seed()

        greenCnt = 0
        yellowCnt = 0
        redCnt = 0
        yellowredCnt = 0

        for i in range(batch_size):
            # get rnd class number
            # print(i)
            classNumber = rnd.randint(1, 4)

            # check if image should be modified to extend data set
            mirrorImage = rnd.randint(0, 1)
            changeContrast = rnd.randint(0, 1)
            changeBrightness = rnd.randint(0, 1)

            if classNumber == 1:
                # Green
                # print("Green")
                minNr = 1
                maxNr = green_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "green" + str(imageNr) + ".jpg"
                pathToFile = self.dir_green_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((1, 0, 0, 0))
                    greenCnt = greenCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 2:
                # Yellow
                # print("Yellow")
                minNr = 1
                maxNr = yellow_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellow" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellow_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((0, 1, 0, 0))
                    yellowCnt = yellowCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 3:
                # Red
                # print("Red")
                minNr = 1
                maxNr = red_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "red" + str(imageNr) + ".jpg"
                pathToFile = self.dir_red_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((0, 0, 1, 0))
                    redCnt = redCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 4:
                # yellowred
                # print("yellowred")
                minNr = 1
                maxNr = yellowred_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellowred" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellowred_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_GRAYSCALE), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((0, 0, 0, 1))
                    yellowredCnt = yellowredCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")

            if mirrorImage == 1:
                batch[i] = cv.flip(batch[i], 1)

            # if changeContrast == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # if changeBrightness == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # flattern image matrix
            batch[i] = np.asarray(batch[i])

        # print("Green: " + str(greenCnt) + " Yellow: " + str(yellowCnt) + " Red: " + str(redCnt)+ " YellowRed: " + str(yellowRedCnt))
        return batch, labels, errorcode
    def load_files_and_get_batch_color_for_classification(self, batch_size):
        # batchsize has to be batchsize%4 == 0
        batch = []
        labels = []
        errorcode = 0

        green_images_cnt = len(os.listdir(self.dir_green_classification))
        yellow_images_cnt = len(os.listdir(self.dir_yellow_classification))
        red_images_cnt = len(os.listdir(self.dir_red_classification))
        yellowred_images_cnt = len(os.listdir(self.dir_yellowred_classification))

        rnd.seed()

        greenCnt = 0
        yellowCnt = 0
        redCnt = 0
        yellowredCnt = 0

        for i in range(batch_size):
            # get rnd class number
            # print(i)
            classNumber = rnd.randint(1, 4)

            # check if image should be modified to extend data set
            mirrorImage = rnd.randint(0, 1)
            changeContrast = rnd.randint(0, 1)
            changeBrightness = rnd.randint(0, 1)

            if classNumber == 1:
                # Green
                # print("Green")
                minNr = 1
                maxNr = green_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "green" + str(imageNr) + ".jpg"
                pathToFile = self.dir_green_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_COLOR), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((1, 0, 0, 0))
                    greenCnt = greenCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 2:
                # Yellow
                # print("Yellow")
                minNr = 1
                maxNr = yellow_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellow" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellow_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_COLOR), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((0, 1, 0, 0))
                    yellowCnt = yellowCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 3:
                # Red
                # print("Red")
                minNr = 1
                maxNr = red_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "red" + str(imageNr) + ".jpg"
                pathToFile = self.dir_red_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_COLOR), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((0, 0, 1, 0))
                    redCnt = redCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")
            if classNumber == 4:
                # yellowred
                # print("yellowred")
                minNr = 1
                maxNr = yellowred_images_cnt
                # print("Max: " + str(maxNr))
                imageNr = rnd.randint(minNr, maxNr)
                filename = "yellowred" + str(imageNr) + ".jpg"
                pathToFile = self.dir_yellowred_classification + "/" + filename
                if os.path.isfile(pathToFile):
                    batch.append(
                        cv.resize(cv.imread(pathToFile, cv.IMREAD_COLOR), (50, 50), interpolation=cv.INTER_CUBIC))
                    labels.append((0, 0, 0, 1))
                    yellowredCnt = yellowredCnt + 1
                else:
                    print("load_files_and_get_batch: Error: the file " + pathToFile + "does not exist")

            if mirrorImage == 1:
                batch[i] = cv.flip(batch[i], 1)

            # if changeContrast == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # if changeBrightness == 1:
            #    batch[i] = cv.flip(batch[i],1)

            # flattern image matrix
            batch[i] = np.asarray(batch[i])

        # print("Green: " + str(greenCnt) + " Yellow: " + str(yellowCnt) + " Red: " + str(redCnt)+ " YellowRed: " + str(yellowRedCnt))
        return batch, labels, errorcode