import os

root = "C:/Users/sakth/OneDrive/Documents/NYU courses/Big Data/Project/"
outputFile = open("C:/Users/sakth/PycharmProjects/bigdataProj/input.txt", "w")
imageNames = os.listdir(root+"dataset/")
for names in imageNames:
    outputFile.write(names+"\n")

outputFile.close()