from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, Layer, Conv2D, MaxPooling2D, UpSampling2D
from keras import losses
from keras.callbacks import LambdaCallback, EarlyStopping
import keras.backend as K

class stockAutoencoder():
    WIDTH = 128
    HEIGHT = 128
    NAME = "stockAutoencoder"

    def __init__(self, optimizer='adagrad'):
        self.optimizer = optimizer
        self._init_model()
        print(self)
        self.autoencoder.summary()

    def __str__(self):
        s = "Stock Autoencoder\n"
        s += "-"*20 + "\n"
        return s

    def encoder_architecture(self, input):
        # encoder
        encoding = Conv2D(8, (4, 4), activation='relu', padding='same')(input)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(16, (4, 4), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(32, (4, 4), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(16, (4, 4), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(8, (4, 4), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(1, (4, 4), activation='sigmoid', padding='same', name='latent_layer')(encoding)
        
        return encoding

    def decoder_architecture(self, input):
        # decoder
        decoding = Conv2D(16, (2, 2), activation='relu', padding='same')(input)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(32, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(64, (4, 4), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(32, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(16, (2, 2), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(decoding)

        return decoding

    def _init_model(self):
        K.clear_session()
        
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=(self.WIDTH, self.HEIGHT,1), name='encoder_input')
        h = self.encoder_architecture(inputs)

        #build decoder model
        outputs = self.decoder_architecture(h)
        
        # instantiate VAE model
        self.autoencoder = Model(inputs, outputs, name='autencoder')
        self.autoencoder.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    def predict(self, x):
        return self.autoencoder.predict(x)

    def train(self, data, epochs=10, batch=32):
        history = None
        try:
            callback = lambda epochs, logs: self._callback_()
            callback = LambdaCallback(on_epoch_end=callback)
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
            history = self.autoencoder.fit(data, data, batch_size=batch, validation_split=0.1, callbacks=[callback, earlystop], epochs=epochs, verbose=1)
        except KeyboardInterrupt:
            pass
        return self.autoencoder, history

    def save_model(self, dir, name=None):
        """Saves model to dir, uses class name by default"""
        if not os.path.isdir(dir):
            os.mkdir(dir)
        if name:
            self.autoencoder.save(os.path.join(dir, name))
        else:
            self.autoencoder.save(os.path.join(dir, self.NAME))
            
    def _callback_(self, evalstocks=['GE', 'MMM'], ext='.pst.npy'):
        for s in evalstocks:
            testDir = "../Data/" + s + "/" + s + "_test/"
            datautils.plot_sample(self.autoencoder, testDir, ext, num=10)

if __name__ == '__main__':
    import numpy as np
    import os
    import stockdatautils as datautils
    
    stocks = ['GE', 'MMM']
    evalstocks = ['GE', 'MMM']
    data = []
    for s in stocks:
        trainDir = "../Data/" + s + "/" + s + "_train/"
        data += [np.array(datautils.load_data_numpy(trainDir, '.pst.npy'))]
    traindata = np.concatenate(data)
    
    data = []
    for s in evalstocks:
        testDir = "../Data/" + s + "/" + s + "_test/"
        data += [np.array(datautils.load_data_numpy(testDir, '.pst.npy'))]
    testdata = np.concatenate(data)
        
    model = stockAutoencoder(optimizer='adagrad')
    try:
        m, history = model.train(traindata, epochs=500, batch=4)
        if history is not None:
            hist = zip(history.history['val_loss'], history.history['val_acc'])
            hist = np.array(hist)
            np.save("history", hist)
    except KeyboardInterrupt:
        pass
    
    model.save_model("../model/", "stockAutoencoder.h5")
    print(model.autoencoder.evaluate(testdata, testdata))
    
    for s in evalstocks:
        testDir = "../Data/" + s + "/" + s + "_test/"
        datautils.plot_sample(model, testDir,'.pst.npy', num=10)
        