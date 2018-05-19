from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras import losses
from keras.layers.normalization import BatchNormalization
import cv2
from Model import Model

class KerasAutoencoder(object):
    WIDTH = 256
    HEIGHT = 256
    
    def __init__(self):
        """
        width - width of picture data
        height - height of picture data
        """
        width = KerasAutoencoder.WIDTH
        height = KerasAutoencoder.HEIGHT
        
        self.model = Sequential()
        
        self.model.add(Dense(128, input_shape=(width * height,), activation = 'relu'))
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('selu'))
        self.model.add(Dense(32))
        self.model.add(BatchNormalization())
        self.model.add(Activation('selu'))
        
        self.model.add(Dense(64))
        self.model.add(Activation('selu'))
        self.model.add(Dense(128))
        self.model.add(Activation('selu'))
        self.model.add(Dense(width * height))
        self.model.add(Activation('selu'))
       
        self.model.compile(optimizer='rmsprop', loss=losses.mean_squared_error)

    def forward(self, x, verbose):
        """
        :param x: an np.array of size (width, height)
        :return: an np.array of size(width, height)
        """
        # Flatten the image
        x.reshape([1,-1])
        x.shape
        return self.model.predict(x, verbose)
    
    def fit(self, data):
        self.model.fit(data, data, batch_size=10, epochs=10,verbose=1,validation_split=0.2)
        return self.model

class KerasModel(Model):
    EPOCHS = 15
    LOG = 10
    BATCH = 10
        
    def __init__(self):
        self.net = KerasAutoencoder()

    def train(self, data):
        epochs = self.EPOCHS
        log = self.LOG
        batch = self.BATCH
        return self.net.fit(data)

    def train_on_generator(self, train, valid):
        pass

    def predict(self, data):
        return self.net.model.predict(data)

if __name__ == '__main__':
    import numpy as np
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    model = KerasModel()

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    list_of_np = []
    DATAPATH="../small_zips/small/ios/"
    TRAIN="training_features"
    
    for fname in os.listdir(DATAPATH+TRAIN):
        if fname[-3:] == 'npz':
            img = rgb2gray(np.load(DATAPATH+TRAIN+"/"+fname)['features'])
            list_of_np.append(img)
    data = np.asarray(list_of_np)
    data = np.reshape(list_of_np, (len(list_of_np),-1))
    
    print(data.shape)
    net = model.train(data)