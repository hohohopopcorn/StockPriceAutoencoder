from keras.models import Model, Sequential
from keras import layers
from keras.layers import Input, Dense, Lambda, Layer, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import losses
from keras.callbacks import LambdaCallback, EarlyStopping
import keras.backend as K

class stockVAE():
    WIDTH = 128
    HEIGHT = 128
    NAME = "stockVAE"

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
        self.latent_size = 32
        self.shrink = 5
        
        encoding = Conv2D(2, (3, 3), activation='relu', padding='same')(input)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(4, (4, 4), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(8, (4, 4), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(4, (4, 4), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        encoding = Conv2D(2, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D((2, 2), padding='same')(encoding)
        
        encoding = Flatten()(encoding)
        encoding = Dense(128, activation='relu')(encoding)
        encoding = Dense(64, activation='relu')(encoding)
        encoding = Dense(self.latent_size, activation='relu')(encoding)
        
        return encoding

    def decoder_architecture(self, input):
        # decoder
        shrink_size = 2**self.shrink
        width = int(self.WIDTH / shrink_size)
        height = int(self.HEIGHT / shrink_size)
        channel = int(self.latent_size / width / height) + 1
        decoding = Dense(width * height * channel, activation='relu')(input)
        decoding = Reshape(target_shape=(width, height, channel))(decoding)
        
        decoding = Conv2D(128, (4, 4), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(64, (4, 4), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(32, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(16, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D((2, 2))(decoding)
        decoding = Conv2D(4, (3, 3), activation='relu', padding='same')(decoding)
        
        decoding = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(decoding)

        return decoding

    def _init_model(self):
        K.clear_session()
        
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=(self.WIDTH, self.HEIGHT,1), name='encoder_input')
        h = self.encoder_architecture(inputs)
        
        z_mean = Dense(self.latent_size, activation='relu', name='mean')(h)
        z_log_dev = Dense(self.latent_size, activation='relu', name='log_deviation')(h)
        
        def sample_z(args):
            mu, log_sigma = args
            batch = K.shape(mu)[0]
            dim = K.int_shape(mu)[1]
            
            eps = K.random_normal(shape=(batch, dim))
            return mu + K.exp(log_sigma / 2) * eps
        
        z = Lambda(sample_z, name='latent_layer')([z_mean, z_log_dev])
        self.encoder = Model(inputs, [z_mean, z_log_dev, z], name='encoder')
        
        #build decoder model
        latent_inputs = Input(shape=(self.latent_size,), name='decoder_input')
        self.decoder = Model(latent_inputs, self.decoder_architecture(latent_inputs), name='decoder')
        
        
        # instantiate VAE model
        reconstructed = self.decoder_architecture(z)
        self.autoencoder = Model(inputs, reconstructed, name='autencoder')
        
        def vae_loss(y_true, y_pred):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
            mu = z_mean
            log_sigma = z_log_dev
            # E[log P(X|z)]
            recon = K.binary_crossentropy(y_true, y_pred) * self.WIDTH * self.HEIGHT
            # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl = - 0.5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
            
            return K.mean(recon + kl)
        
        self.autoencoder.compile(optimizer=self.optimizer, loss=vae_loss, metrics=['accuracy'])
    
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
        
    model = stockVAE(optimizer='adagrad')
    try:
        m, history = model.train(traindata, epochs=500, batch=4)
        if history is not None:
            hist = zip(history.history['val_loss'], history.history['val_acc'])
            hist = np.array(hist)
            np.save("history", hist)
    except KeyboardInterrupt:
        pass
    
    model.save_model("../model/", "stockVAE.h5")
    print(model.autoencoder.evaluate(testdata, testdata))
    
    for s in evalstocks:
        testDir = "../Data/" + s + "/" + s + "_test/"
        datautils.plot_sample(model, testDir,'.pst.npy', num=10)
        