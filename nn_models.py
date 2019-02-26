# -*- coding: utf-8 -*-
"""
    Functions to deal with neural network models created in keras.
"""
from scipy import sparse as sp_sparse
import numpy as np
import itertools
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras import models, layers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint

NN_MODEL_PATH={"DENSE_MODEL": "dense_model.h5",
               "CNN_MODEL": "cnn_model.h5"}

class NN_Data_Prepare():
    """
        The 'NN_Data_prepare' class converts text data and its labels to numeric format that can be used as input for a neural network. 
    
        First, a dictionary of the most common words is created. 
        Based on the dictionary, one of two transformations can be applied to the text. 
        The goal of the first one is to get one-hot representation of the text.
        The goal of the second one is to get bag-of-words representation of the text padded to the predefined length.
    """
    
    def __init__(self, dict_size=10000):
        """
            Specify dictionary size, e.g. number of the most common words to be included in the dictionary. 
            
            dict_size: an integer 
        """
        
        self.dict_size = dict_size
  
    def create_dict(self, x_train):
        """
            Create dictionary of the most common words in the corpus 'x_train'.
            
            x_train: list of strings
        """
        # create dictionary of all words from corpus with their counts
        words_counts = Counter(list(itertools.chain.from_iterable(list(map(lambda x: x.split(), x_train)))))
        # select top of the most common words
        common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
        words_to_index = {word[0]: index for index, word in enumerate(common_words[0:self.dict_size])}
        self.words_to_index = words_to_index
  
    def text_ohe(self, text):
        """
            Create a one-hot representation of the input text.
          
            text: a string
          
            return: a vector which is a one-hot representation of 'text'
        """
        result_vector = np.zeros(self.dict_size)
        for word in text.split():
            if word in self.words_to_index.keys():
                result_vector[self.words_to_index[word]] = 1
        return result_vector
  
    def set_ohe(self, x):
        """
            Create one-hot representation for each item from the input list.
            
            x: list of strings
          
            return: list of vectors that are one-hot representation of items from 'x'
        """
        vectorized_representation = sp_sparse.vstack([sp_sparse.csr_matrix(self.text_ohe(text)) for text in x])
        return vectorized_representation 
  
    def text_to_numbers(self, text):
        """
            Create bag-of-words representation of the input text.
            Unknown words, i.e. words that are not from the dictionary, are thrown away.
            
            text: a string
            return: a vector which is bag-of-words representation of 'text'.
        """
        result_vector = np.full(len(text.split()), fill_value=-1)
        for i in range(len(text.split())):
            word = text.split()[i]
            if word in self.words_to_index.keys():
                result_vector[i] = self.words_to_index[word]
        result_vector = result_vector[result_vector > 0]
        return result_vector
    
    def set_to_numbers(self, x):
        """
            Create bag-of-words representation for each item from the input list.
          
            x: list of strings
            
            return: list of vectors that are bag-of-words representations of items from 'x'
        """
        result_vector = [self.text_to_numbers(text) for text in x]
        return result_vector
  
    def padded_set(self, x, maxlen=1000):
        """
            Pad sequences from 'x' to the same length of 'maxlen'. 
            
            x: list
            maxlen: integer
            
            return: list of padded sequences
        """
    
        result_vector = pad_sequences(x, maxlen=maxlen, value=len(self.words_to_index.keys())+1)
        return result_vector
    
    def vectorized_labels(self, labels):
        """
            Convert 'labels' to float numpy array.
            
            labels: list
            
            return: numpy array
        """
        return np.asarray(labels).astype('float32')
    
class NN_Model():
    
    """
        The 'NN_Models' class creates instance of 'keras.engine.sequential.Sequential'.
        
    """  
    
    def dense_model(self, hidden_layers=2, output=16, input_size=10000, dropout=0.4):
        """
            Create an instance of 'keras.engine.sequential.Sequential' with dense layers only.
            
            hidden_layers: integer, number of dense layers with dropouts
            output: integer, dimensionality of output space 
            input_size: integer, input shape for the first layer
            dropout: float, dropout rate
            
            return: an instance of 'keras.engine.sequential.Sequential'
        """
        
        dense_model = models.Sequential()
        dense_model.add(layers.Dense(output, activation='relu', input_shape=(input_size,),\
                                     kernel_initializer='he_normal'))
        dense_model.add(layers.Dropout(dropout))
        
        for i in range(0, hidden_layers):
            dense_model.add(layers.Dense(output, activation='relu', kernel_initializer='he_normal'))
            dense_model.add(layers.Dropout(dropout))
            
        dense_model.add(layers.Dense(1, activation='sigmoid'))
        
        return dense_model
    
    def cnn_model(self, dict_size=10000, emb_out=100, maxlen=500, hid_out=250, dropout=0.3,\
            filters=200, kernel_size=2):
        """
            Create an instance of 'keras.engine.sequential.Sequential' with Conv1D layer
            
            dict_size: number of words in the dictionary
            emb_out: dimansionality of embeddings
            maxlen: length of inputs
            filters: number of filters
            kernel_size: kernel size
            
            return: an instance of 'keras.engine.sequential.Sequential'
            
        """
        
        model = models.Sequential()
        model.add(layers.Embedding(dict_size, emb_out, input_length=maxlen))
        model.add(layers.Dropout(dropout))
        model.add(layers.Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(hid_out, activation='relu', kernel_initializer='he_normal'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model
    
    
class NN_Train_Predict():
    """
        Wrapper to fit keras neural network model and generate predictions. 
    """
    
    def compile_and_fit_model(self, model, x_train, y_train, x_val, y_val, epochs=20, \
                            path_to_save="best_model.h5", loss='binary_crossentropy', \
                            lr=0.001, batch_size=32, shuffle=True):
        """
            Compile and fit the model, save it to file.
            
            model: model to fit
            x_train, y_train: train samples and labels
            x_val, y_val: validation samples and labels
            epochs: number of epochs
            loss: loss function
            lr: learning rate
            batch_size: batch size
            shuffle: boolean, whether to shuffle input data  
            
            returns: record of training loss and metrics values
        """
        
        model.compile(optimizer='adam', loss=loss, metrics=[metrics.binary_accuracy])
        callback = [EarlyStopping(monitor='val_loss', patience=2),
                    ModelCheckpoint(filepath=path_to_save, monitor='val_loss', save_best_only=True)]
    
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        shuffle=shuffle, validation_data=(x_val, y_val), callbacks=callback)
        return history.history

    def model_predict(self, model, x_test):
        """
            Generate output predictions.
            x_test: test data
            
            return: ndarray of output probabilities
        """
        return model.predict(x_test)
    
    
    def model_predict_classes(self, model, x_test):
        """
            Generate class predictions.
            x_test: test data
            
            return: ndarray of class predictions
        """
        return model.predict_classes(x_test)
    
    def model_evaluate(self, model, x_test, y_test):
        """
            Generate output predictions.
            x_test, y_test: test data and labels
            
            return: loss and metric values
        """
        return model.evaluate(x_test, y_test)
      


