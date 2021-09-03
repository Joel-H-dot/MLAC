from tensorflow.keras import Model, layers, regularizers,optimizers,utils,models, Input
import tensorflow as tf
import kerastuner as kt
import pathlib
import os.path
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class autoencoder_hyperband(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_dim=0):
        self.encoder_dim = encoder_dim
        self.epochs = 100
        self.patience = 10
        self.factor = 3

    def build_autoencoder(self,hp):

        inputs = Input(shape=(self.X.shape[1],))

        #### Encoder block

        hp_layers_encoder = hp.Int('layers encoder', min_value=0, max_value=8, step=1)

        hp_regularisation_encoder = hp.Float('kernel_regularizer encoder', max_value=1e-1, min_value=1e-12, sampling='log')

        hp_units_encoder = list()
        for j in range(0, hp_layers_encoder):
            hp_units_encoder.append(hp.Int('encoder units L' + str(j + 1), min_value=16, max_value=256, step=32))

        x_encoder = inputs

        for i in range(0, hp_layers_encoder):
            x_encoder = layers.Dense(units=hp_units_encoder[i], kernel_regularizer=regularizers.l2(hp_regularisation_encoder),
                                       activation='relu')(x_encoder)

        outputs_encoder = layers.Dense(self.encoder_dim, name='encoder_output')(x_encoder)

        #### Decoder block

        hp_layers_decoder = hp.Int('layers decoder', min_value=0, max_value=8, step=1)

        hp_regularisation_decoder = hp.Float('kernel_regularizer decoder', max_value=1e-1, min_value=1e-12, sampling='log')

        hp_units_decoder = list()
        for j in range(0, hp_layers_decoder):
            hp_units_decoder.append(hp.Int('decoder units L' + str(j + 1), min_value=16, max_value=256, step=32))

        x_decoder = outputs_encoder

        for i in range(0, hp_layers_decoder):
            x_decoder = layers.Dense(units=hp_units_decoder[i],
                                           kernel_regularizer=regularizers.l2(hp_regularisation_decoder),
                                           activation='relu')(x_decoder)


        outputs = layers.Dense(self.X.shape[1])(x_decoder)

        model = Model(inputs=inputs, outputs=outputs)

        hp_learning_rate = hp.Float('learning_rate', max_value=1, min_value=1e-5, sampling='log')
        hp_batches = hp.Choice('batch', values=[1, 8, 16, 32])
        hp_optimizer = hp.Choice('search algorithm', values=['Adam', 'RMSprop', 'GD'])
        if hp_optimizer == 'Adam':
            model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mse'])
        if hp_optimizer == 'RMSprop':
            model.compile(optimizer=optimizers.RMSprop(learning_rate=hp_learning_rate), loss='mse', metrics=['mse'])
        if hp_optimizer == 'GD':
            model.compile(optimizer=optimizers.SGD(learning_rate=hp_learning_rate), loss='mse',
                          metrics=['mse'])

        return model


    def fit(self,X,y):  #load encoder if it exists! for current classifier
        self.X=X

        try:
            self.hypermodel = models.load_model('ML_data\CD data\Models\AE_model_'+str(self.encoder_dim)+'.h5')
            #raise ValueError('Cannot load model')
        except:
            path_dir = 'C:/'
            path_dir = os.path.normpath(path_dir)

            #### check to see if checkpoint directory already exists
            L = 1
            while True:
                file_long = os.path.normpath(path_dir + '\HP_checkpoints_AE_' + str(L))
                if os.path.isdir(file_long):
                    L = L + 1
                else:
                    break

            epochs = self.epochs
            tuner = kt.Hyperband(self.build_autoencoder,
                                 objective='val_mse',
                                 max_epochs=epochs,
                                 factor=self.factor,
                                 directory=path_dir,
                                 project_name='HP_checkpoints_AE_'+str(L),
                                 allow_new_entries=True,
                                 tune_new_entries=True,
                                 overwrite=False)

            # tuner = kt.RandomSearch(self.build_model,
            #                      objective='val_mse',
            #                      max_trials=200,
            #                      directory='my_dir',
            #                      project_name='HP_search',
            #                      allow_new_entries=True,
            #                      tune_new_entries=True,
            #                      overwrite=False)

            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=self.patience)

            tuner.search(X, X, validation_split=0.2, callbacks=[stop_early])

            best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

            model = tuner.hypermodel.build(best_hyperparameters)
            history = model.fit(X, X, epochs=epochs, batch_size=8, validation_split=0.2)

            val_mse_per_epoch = history.history['val_mse']
            best_epoch = val_mse_per_epoch.index(min(val_mse_per_epoch)) + 1
            print('Best epoch: %d' % (best_epoch,))

            self.hypermodel = tuner.hypermodel.build(best_hyperparameters)

            # print(self.encoder_dim)
            # for layer in self.hypermodel.layers:
            #     print(layer.name)
            # self.hypermodel.get_layer("encoder_output")
            # Retrain the model
            self.hypermodel.fit(X, X, epochs=best_epoch, batch_size=8, validation_data=None)
            self.hypermodel.save('ML_data\CD data\Models\AE_model_' + str(self.encoder_dim) + '.h5')

            try:
                import shutil
                shutil.rmtree(file_long)
            except:
                print('File Already deleted')

        return self

    def transform(self,data,y=None):

        encoder = Model(inputs=self.hypermodel.input, outputs=self.hypermodel.get_layer("encoder_output").output)
        encoder_output = encoder.predict(data)

        return encoder_output

class FCN_hyperband(BaseEstimator, ClassifierMixin):
    def __init__(self, Threshold=0, epochs = 5, patience=2, factor=3):
        self.Threshold = Threshold
        self.epochs = epochs
        self.patience = patience
        self.factor = factor

    def build_CNN(self,hp):

        inputs = Input(shape=(self.X.shape[1],))

        hp_layers = hp.Int('layers', min_value=0, max_value=8, step=1)

        hp_regularisation = hp.Float('kernel_regularizer', max_value=1e-1, min_value=1e-12, sampling='log')


        hp_filters = hp.Int('filters', min_value=0, max_value=8, step=1)

        hp_filter_kernel=  hp.Int('K', min_value=1, max_value=6, step=1)

        hp_units = list()
        for j in range(0, hp_layers):
            hp_units.append(hp.Int('units L' + str(j + 1), min_value=16, max_value=256, step=32))

        x = inputs

        # x = layers.Conv1D(filters = hp_filters, kernel_size=hp_filter_kernel, activation='relu')(x)
        # x = layers.Flatten()(x)

        for i in range(0, hp_layers):
            x = layers.Dense(units=hp_units[i], kernel_regularizer=regularizers.l2(hp_regularisation),activation='relu')(x)


        x = layers.Flatten()(x)
        outputs = layers.Dense(units = 1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        hp_learning_rate = hp.Float('learning_rate', max_value=1, min_value=1e-5, sampling='log')
        hp_batches = hp.Choice('batch', values=[1, 8, 16, 32])
        hp_optimizer = hp.Choice('search algorithm', values=['Adam', 'RMSprop', ])#'GD'])
        if hp_optimizer == 'Adam':
            model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
        if hp_optimizer == 'RMSprop':
            model.compile(optimizer=optimizers.RMSprop(learning_rate=hp_learning_rate), loss='binary_crossentropy',
                          metrics=['binary_accuracy'])#, loss='mse', metrics=['mse'])
        if hp_optimizer == 'GD':
            model.compile(optimizer=optimizers.SGD(learning_rate=hp_learning_rate), loss='binary_crossentropy',
                          metrics=['binary_accuracy'])

        return model


    def fit(self,X,y):  #load encoder if it exists! for current classifier
        self.X=X
        self.y = y


        try:
            #self.hypermodel = load_model('ML_data\CD data\Models\CNN_model_'+str(self.encoder_dim)+'.h5')
            raise ValueError('Cannot load model')
        except:
            s = str(pathlib.Path(__file__).parent.absolute())

            path_dir = 'C:/'
            path_dir = os.path.normpath(path_dir)


            #### check to see if checkpoint directory already exists
            L = 1
            while True:
                file_long = os.path.normpath(path_dir + '\HP_checkpoints_CNN_' + str(L))
                if os.path.isdir(file_long):
                    L = L + 1
                else:
                    break

            epochs = self.epochs
            tuner = kt.Hyperband(self.build_CNN,
                                 objective='val_binary_accuracy',
                                 max_epochs=epochs,
                                 factor=self.factor,
                                 directory=path_dir,
                                 project_name='HP_checkpoints_CNN_'+str(L),
                                 allow_new_entries=True,
                                 tune_new_entries=True,
                                 overwrite=False)

            # tuner = kt.RandomSearch(self.build_model,
            #                      objective='val_mse',
            #                      max_trials=200,
            #                      directory='my_dir',
            #                      project_name='HP_search',
            #                      allow_new_entries=True,
            #                      tune_new_entries=True,
            #                      overwrite=False)

            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=self.patience)

            tuner.search(self.X, self.y, validation_split=0.2, callbacks=[stop_early])

            best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

            model = tuner.hypermodel.build(best_hyperparameters)
            history = model.fit(self.X, self.y, epochs=epochs, batch_size=8, validation_split=0.2)

            val_mse_per_epoch = history.history['val_binary_accuracy']
            best_epoch = val_mse_per_epoch.index(max(val_mse_per_epoch)) + 1
            print('Best epoch: %d' % (best_epoch,))

            self.hypermodel = tuner.hypermodel.build(best_hyperparameters)

            # print(self.encoder_dim)
            # for layer in self.hypermodel.layers:
            #     print(layer.name)
            # self.hypermodel.get_layer("encoder_output")
            # Retrain the model
            self.hypermodel.fit(self.X, self.y, epochs=best_epoch, batch_size=8, validation_data=None)

            try:
                import shutil
                shutil.rmtree(file_long)
            except:
                print('File Already deleted')
            #self.hypermodel.save('ML_data\CD data\Models\model_' + str(self.encoder_dim) + '.h5')
        return self

    def predict(self,data):
        predicted=self.hypermodel.predict(data)

        for pp in range(0, len(predicted.squeeze())):
            if predicted[pp] > self.Threshold:
                predicted[pp] = 1
            else:
                predicted[pp] = 0

        return predicted
