import sys
import os
import numpy as np
from sklearn import preprocessing
from skimage.transform import resize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg', warn=False)
from skimage import color
from math import log
import win_unicode_console
win_unicode_console.enable()

history = 20
forecast = 5
FACTORS = 2
width = 128
height = width
stocks = ["MMM", "GE"]
modes = ["train", "test"]

# def normalize(arr):
#     maximum = []
#     minimum = []
#     for i in range(len(arr[0])):
#         temp = [arr[j][i] for j in range(len(arr))]
#         maximum += [max(temp)]
#         minimum += [min(temp)]
#     copy = []
#     for j in range(len(arr)):
#         copy.append([(arr[j][i] - minimum[i]) / (maximum[i] - minimum[i])
#                      for i in range(len(arr[j]))])
#     return copy

def normalize(arr):
    copy = []
    for j in range(len(arr)):
        copy.append([log(arr[j][i] / arr[0][i]) for i in range(len(arr[j]))])
    return copy

def plotting(x):
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    fig.add_subplot(1,1,1)
    plt.plot(x, color="black", linewidth=5.0*height/100)
    plt.axis('off')
    fig.canvas.draw()

    mplimage = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    mplimage = mplimage.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    gray_image = color.rgb2gray(mplimage)
    gray_image = resize(gray_image, (width, width, 1))
    plt.close(fig)
    
    return gray_image

for s in stocks:
    for m in modes:
        directory = os.path.abspath('../Data/'+ s + '/' + s + '_' + m)
        dataFile = open(s + '_' + m + ".txt")
        dates = []
        
        print("Extracting " + str(dataFile.name))
        
        for line in dataFile:
            doubles = line.split()
            if len(doubles) == FACTORS:
                dates.append([float(doubles[i]) for i in range(len(doubles)-1)])
        
        length = len(dates) - history - forecast
        
        for i in range(length):
            temp = normalize(dates[i:i+history+forecast])
            x = plotting(np.array(temp[:history]).flatten())
            y = plotting(np.array(temp[history:]).flatten())
            print(dataFile.name, i, length,  x.shape, y.shape)
            np.save(directory + "/" + s + "_" + m + str(i) + ".pst", x)
            np.save(directory + "/" + s + "_" + m + str(i) + ".ftr", y)
        print("Done Extracting " + str(dataFile.name))
