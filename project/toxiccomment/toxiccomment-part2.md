[<-PREV](toxiccomment.md)

# Toxic Comment Classification Part 2 
# Deep Learning NLP with RNN
1. Set up
2. Text preprocessing
    - Clean the text
    - Tokenize
    - Pad
3. Train Test split
4. Word embedding: GloVe
5. Build, Fit/Train, Predict, and Evaluate models
6. Compare the results

# 1. Set up
## 1.1 Import libraries


```python
pip install clean-text
```

    Collecting clean-text
      Downloading clean_text-0.4.0-py3-none-any.whl (9.8 kB)
    Collecting ftfy<7.0,>=6.0
      Downloading ftfy-6.0.3.tar.gz (64 kB)
    [K     |████████████████████████████████| 64 kB 917 kB/s eta 0:00:01
    [?25hRequirement already satisfied: emoji in /opt/conda/lib/python3.7/site-packages (from clean-text) (1.2.0)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from ftfy<7.0,>=6.0->clean-text) (0.2.5)
    Building wheels for collected packages: ftfy
      Building wheel for ftfy (setup.py) ... [?25ldone
    [?25h  Created wheel for ftfy: filename=ftfy-6.0.3-py3-none-any.whl size=41913 sha256=c5ef0cd989ce9bab8924750fd283a569a4c4bb8d1942db862986c0cbe161dd9b
      Stored in directory: /root/.cache/pip/wheels/19/f5/38/273eb3b5e76dfd850619312f693716ac4518b498f5ffb6f56d
    Successfully built ftfy
    Installing collected packages: ftfy, clean-text
    Successfully installed clean-text-0.4.0 ftfy-6.0.3
    Note: you may need to restart the kernel to use updated packages.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
print('Tensorflow version: ', tf.__version__)
from tensorflow import keras

from cleantext import clean
```

    Tensorflow version:  2.4.1


## 1.2 Configuration - TPU


```python
def setup_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU: ', tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy() # for CPU or GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
    return strategy

strategy = setup_accelerator()
```

    Running on TPU:  grpc://10.0.0.2:8470
    Number of replicas: 8


## 1.3 Set up directories


```python
input_dir = '../input/toxiccomment-part1/'
output_dir = ''
```

## 1.4 Load the data


```python
df = pd.read_csv(input_dir + 'preprocessed.csv')
```


```python
print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32450 entries, 0 to 32449
    Data columns (total 2 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   comment_text  32450 non-null  object
     1   Toxic         32450 non-null  int64 
    dtypes: int64(1), object(1)
    memory usage: 507.2+ KB
    None





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment_text</th>
      <th>Toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hey... what is it..\n@ | talk .\nWhat is it......</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bye! \n\nDon't look, come or think of comming ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>You are gay or antisemmitian? \n\nArchangel WH...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FUCK YOUR FILTHY MOTHER IN THE ASS, DRY!</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 1.5 Configuration


```python
n_classes = 1
SEED = 42
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
print(f'Batch size: {BATCH_SIZE}')

EPOCHS = 20
```

    Batch size: 128


## 1.6 Define X, y


```python
X = df['comment_text']
y = df['Toxic']
```

# 2. Text preprocessing

## 2.1 Clean the text


```python
def cleaning(text):
    return clean(text, no_line_breaks=True, no_urls=True, no_punct=True)

cleaned_X = np.vectorize(cleaning)(X)
```


```python
idx=10
print('----------Original text----------')
print(X[idx])
print('----------Cleaned text----------')
print(cleaned_X[idx])
```

    ----------Original text----------
    Why can't you believe how fat Artie is? Did you see him on his recent appearence on the Tonight Show with Jay Leno? He looks absolutely AWFUL! If I had to put money on it, I'd say that Artie Lange is a can't miss candidate for the 2007 Dead pool!   
    
      
    Kindly keep your malicious fingers off of my above comment, . Everytime you remove it, I will repost it!!!
    ----------Cleaned text----------
    why cant you believe how fat artie is did you see him on his recent appearence on the tonight show with jay leno he looks absolutely awful if i had to put money on it id say that artie lange is a cant miss candidate for the 2007 dead pool kindly keep your malicious fingers off of my above comment everytime you remove it i will repost it


## 2.2 Tokenize


```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

t = Tokenizer()
t.fit_on_texts(cleaned_X)
vocab_size = len(t.word_index) + 1
encoded_X = t.texts_to_sequences(cleaned_X)
```

## 2.3 Pad


```python
len_X = [len(x) for x in encoded_X ]

plt.figure(figsize = (6, 4))
sns.histplot(data=len_X, bins = np.arange(0, 410, 10))
plt.xlabel('The length of X');
```


    
![png](output_22_0.png)
    



```python
maxlen = 150
padded_X = pad_sequences(encoded_X, maxlen=maxlen, padding='post')
```


```python
idx = 1
print('----------Original----------')
print(X[idx])
print('----------Tokenized----------')
print(encoded_X[idx])
print('----------Padded----------')
print(padded_X[idx])
```

    ----------Original----------
    Hey... what is it..
    @ | talk .
    What is it... an exclusive group of some WP TALIBANS...who are good at destroying, self-appointed purist who GANG UP any one who asks them questions abt their ANTI-SOCIAL and DESTRUCTIVE (non)-contribution at WP?
    
    Ask Sityush to clean up his behavior than issue me nonsensical warnings...
    ----------Tokenized----------
    [197, 34, 8, 11, 50, 34, 8, 11, 29, 4706, 533, 7, 63, 832, 33755, 14, 111, 43, 3441, 6873, 33756, 51, 3124, 60, 74, 54, 51, 4985, 97, 397, 23129, 104, 10694, 6, 4707, 33757, 43, 832, 277, 23130, 3, 1457, 60, 65, 930, 105, 348, 23, 6874, 1203]
    ----------Padded----------
    [  197    34     8    11    50    34     8    11    29  4706   533     7
        63   832 33755    14   111    43  3441  6873 33756    51  3124    60
        74    54    51  4985    97   397 23129   104 10694     6  4707 33757
        43   832   277 23130     3  1457    60    65   930   105   348    23
      6874  1203     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0     0     0]


# 3. Train Test splilt


```python
X_train, X_test, y_train, y_test = train_test_split(padded_X, y, test_size=0.2, random_state=SEED)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED)
n_train = X_train.shape[0]
n_valid = X_valid.shape[0]
n_test = X_test.shape[0]
print(f'The size of train set: {n_train}')
print(f'The size of valid set: {n_valid}')
print(f'The size of test set: {n_test}')
```

    The size of train set: 25960
    The size of valid set: 3245
    The size of test set: 3245


# 4. Word Embedding: GloVe


```python
embeddings_index = {}
f = open('../input/glove6b300dtxt/glove.6B.300d.txt','r',encoding='utf-8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
```

    Loaded 400000 word vectors.



```python
embed_size = 300
embedding_matrix = np.zeros((vocab_size, embed_size))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```

# 5. Build, Compile, Train, Evaluate models


```python
def plot_train_history(history):
    
    fig, ax = plt.subplots(2, 1, figsize=(7, 8))
    x = np.arange(1, len(history.history['loss'])+1)

    ax[0].plot(x, history.history['loss'])
    ax[0].plot(x, history.history['val_loss'])
    ax[0].set_xlabel('epochs')
    ax[0].set_xticks(x)
    ax[0].set_ylabel('loss', fontsize=15)
    ax[0].legend(['train', 'val'], loc='upper right')

    ax[1].plot(x, history.history['accuracy'])
    ax[1].plot(x, history.history['val_accuracy'])
    ax[1].set_xlabel('epochs')
    ax[1].set_xticks(x)
    ax[1].set_ylabel('accuracy', fontsize=15)
    ax[1].legend(['train', 'val'], loc='lower right');
```


```python
from keras.models import Sequential
from keras.layers import RNN, SimpleRNN, GRU, LSTM, Bidirectional, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
```


```python
results = {'GRU': None, 'LSTM': None, 'BiLSTM': None}
```

## 5.1 GRU


```python
keras.backend.clear_session()

with strategy.scope():
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))
    model.add(GRU(256))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

FILEPATH = output_dir + "toxiccomment_gru.h5"
ckp = ModelCheckpoint(FILEPATH, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1, min_delta = 1e-4, min_lr = 1e-8, mode = 'min', cooldown=1)
es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min', restore_best_weights = True, verbose = 1)

history = model.fit(X_train, 
                    y_train, 
                    steps_per_epoch= n_train // BATCH_SIZE,
                    validation_data=(X_valid, y_valid),
                    validation_batch_size=n_valid,
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS,
                    callbacks=[ckp, rlr, es],
                    verbose=1)

plot_train_history(history)
test_result = model.evaluate(X_test, y_test, batch_size=strategy.num_replicas_in_sync)
results['GRU'] = test_result
print(f"Test result: {test_result}")
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 150, 300)          23203800  
    _________________________________________________________________
    gru (GRU)                    (None, 256)               428544    
    _________________________________________________________________
    dense (Dense)                (None, 1)                 257       
    =================================================================
    Total params: 23,632,601
    Trainable params: 428,801
    Non-trainable params: 23,203,800
    _________________________________________________________________
    Epoch 1/20
    202/202 [==============================] - 10s 29ms/step - loss: 0.6746 - accuracy: 0.5404 - val_loss: 0.6458 - val_accuracy: 0.5948
    
    Epoch 00001: val_loss improved from inf to 0.64584, saving model to toxiccomment_gru.h5
    Epoch 2/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.6598 - accuracy: 0.5637 - val_loss: 0.6481 - val_accuracy: 0.5744
    
    Epoch 00002: val_loss did not improve from 0.64584
    Epoch 3/20
    202/202 [==============================] - 5s 22ms/step - loss: 0.5254 - accuracy: 0.6811 - val_loss: 0.2196 - val_accuracy: 0.9119
    
    Epoch 00003: val_loss improved from 0.64584 to 0.21957, saving model to toxiccomment_gru.h5
    Epoch 4/20
    202/202 [==============================] - 4s 22ms/step - loss: 0.2186 - accuracy: 0.9100 - val_loss: 0.2038 - val_accuracy: 0.9233
    
    Epoch 00004: val_loss improved from 0.21957 to 0.20384, saving model to toxiccomment_gru.h5
    Epoch 5/20
    202/202 [==============================] - 4s 22ms/step - loss: 0.1941 - accuracy: 0.9218 - val_loss: 0.1992 - val_accuracy: 0.9220
    
    Epoch 00005: val_loss improved from 0.20384 to 0.19923, saving model to toxiccomment_gru.h5
    Epoch 6/20
    202/202 [==============================] - 5s 22ms/step - loss: 0.1738 - accuracy: 0.9311 - val_loss: 0.2145 - val_accuracy: 0.9134
    
    Epoch 00006: val_loss did not improve from 0.19923
    Epoch 7/20
    202/202 [==============================] - 4s 22ms/step - loss: 0.1601 - accuracy: 0.9381 - val_loss: 0.1970 - val_accuracy: 0.9254
    
    Epoch 00007: val_loss improved from 0.19923 to 0.19696, saving model to toxiccomment_gru.h5
    Epoch 8/20
    202/202 [==============================] - 5s 22ms/step - loss: 0.1374 - accuracy: 0.9478 - val_loss: 0.2032 - val_accuracy: 0.9251
    
    Epoch 00008: val_loss did not improve from 0.19696
    Epoch 9/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1161 - accuracy: 0.9548 - val_loss: 0.2275 - val_accuracy: 0.9208
    
    Epoch 00009: val_loss did not improve from 0.19696
    Epoch 10/20
    202/202 [==============================] - 5s 22ms/step - loss: 0.0924 - accuracy: 0.9669 - val_loss: 0.2476 - val_accuracy: 0.9183
    
    Epoch 00010: val_loss did not improve from 0.19696
    
    Epoch 00010: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 11/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.0722 - accuracy: 0.9763 - val_loss: 0.2713 - val_accuracy: 0.9208
    
    Epoch 00011: val_loss did not improve from 0.19696
    Epoch 12/20
    202/202 [==============================] - 5s 22ms/step - loss: 0.0467 - accuracy: 0.9865 - val_loss: 0.2907 - val_accuracy: 0.9165
    
    Epoch 00012: val_loss did not improve from 0.19696
    Restoring model weights from the end of the best epoch.
    Epoch 00012: early stopping
    406/406 [==============================] - 7s 14ms/step - loss: 0.2115 - accuracy: 0.9208
    Test result: [0.21147219836711884, 0.9208012223243713]



    
![png](images/output_35_1.png)
    


## 5.2 LSTM


```python
keras.backend.clear_session()

with strategy.scope():
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))
    model.add(LSTM(256))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

FILEPATH = output_dir + "toxiccomment_lstm.h5"
ckp = ModelCheckpoint(FILEPATH, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1, min_delta = 1e-4, min_lr = 1e-8, mode = 'min', cooldown=1)
es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min', restore_best_weights = True, verbose = 1)

history = model.fit(X_train, 
                    y_train, 
                    steps_per_epoch= n_train // BATCH_SIZE,
                    validation_data=(X_valid, y_valid),
                    validation_batch_size=n_valid,
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS,
                    callbacks=[ckp, rlr, es],
                    verbose=1)

plot_train_history(history)
test_result = model.evaluate(X_test, y_test, batch_size=strategy.num_replicas_in_sync)
results['LSTM'] = test_result
print(f"Test result: {test_result}")
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 150, 300)          23203800  
    _________________________________________________________________
    lstm (LSTM)                  (None, 256)               570368    
    _________________________________________________________________
    dense (Dense)                (None, 1)                 257       
    =================================================================
    Total params: 23,774,425
    Trainable params: 570,625
    Non-trainable params: 23,203,800
    _________________________________________________________________
    Epoch 1/20
    202/202 [==============================] - 10s 29ms/step - loss: 0.6642 - accuracy: 0.5619 - val_loss: 0.6607 - val_accuracy: 0.6419
    
    Epoch 00001: val_loss improved from inf to 0.66075, saving model to toxiccomment_lstm.h5
    Epoch 2/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.5424 - accuracy: 0.7374 - val_loss: 0.3086 - val_accuracy: 0.8847
    
    Epoch 00002: val_loss improved from 0.66075 to 0.30857, saving model to toxiccomment_lstm.h5
    Epoch 3/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.3398 - accuracy: 0.8670 - val_loss: 0.2752 - val_accuracy: 0.8974
    
    Epoch 00003: val_loss improved from 0.30857 to 0.27525, saving model to toxiccomment_lstm.h5
    Epoch 4/20
    202/202 [==============================] - 4s 22ms/step - loss: 0.2837 - accuracy: 0.8853 - val_loss: 0.2836 - val_accuracy: 0.8807
    
    Epoch 00004: val_loss did not improve from 0.27525
    Epoch 5/20
    202/202 [==============================] - 4s 22ms/step - loss: 0.2445 - accuracy: 0.9009 - val_loss: 0.2138 - val_accuracy: 0.9162
    
    Epoch 00005: val_loss improved from 0.27525 to 0.21378, saving model to toxiccomment_lstm.h5
    Epoch 6/20
    202/202 [==============================] - 4s 22ms/step - loss: 0.2132 - accuracy: 0.9132 - val_loss: 0.2050 - val_accuracy: 0.9171
    
    Epoch 00006: val_loss improved from 0.21378 to 0.20496, saving model to toxiccomment_lstm.h5
    Epoch 7/20
    202/202 [==============================] - 5s 22ms/step - loss: 0.1975 - accuracy: 0.9214 - val_loss: 0.1983 - val_accuracy: 0.9242
    
    Epoch 00007: val_loss improved from 0.20496 to 0.19832, saving model to toxiccomment_lstm.h5
    Epoch 8/20
    202/202 [==============================] - 5s 22ms/step - loss: 0.1849 - accuracy: 0.9282 - val_loss: 0.2089 - val_accuracy: 0.9186
    
    Epoch 00008: val_loss did not improve from 0.19832
    Epoch 9/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1628 - accuracy: 0.9365 - val_loss: 0.2085 - val_accuracy: 0.9199
    
    Epoch 00009: val_loss did not improve from 0.19832
    Epoch 10/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1487 - accuracy: 0.9439 - val_loss: 0.2090 - val_accuracy: 0.9239
    
    Epoch 00010: val_loss did not improve from 0.19832
    
    Epoch 00010: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 11/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1342 - accuracy: 0.9510 - val_loss: 0.2183 - val_accuracy: 0.9239
    
    Epoch 00011: val_loss did not improve from 0.19832
    Epoch 12/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1125 - accuracy: 0.9603 - val_loss: 0.2243 - val_accuracy: 0.9220
    
    Epoch 00012: val_loss did not improve from 0.19832
    Restoring model weights from the end of the best epoch.
    Epoch 00012: early stopping
    406/406 [==============================] - 8s 14ms/step - loss: 0.2197 - accuracy: 0.9094
    Test result: [0.21967385709285736, 0.9093990325927734]



    
![png](images/output_37_1.png)
    


## 5.3 Bidirectional LSTM


```python
keras.backend.clear_session()

with strategy.scope():
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

FILEPATH = output_dir + "toxiccomment_bilstm.h5"
ckp = ModelCheckpoint(FILEPATH, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1, min_delta = 1e-4, min_lr = 1e-8, mode = 'min', cooldown=1)
es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min', restore_best_weights = True, verbose = 1)

history = model.fit(X_train, 
                    y_train, 
                    steps_per_epoch= n_train // BATCH_SIZE,
                    validation_data=(X_valid, y_valid),
                    validation_batch_size=n_valid,
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS,
                    callbacks=[ckp, rlr, es],
                    verbose=1)

plot_train_history(history)
test_result = model.evaluate(X_test, y_test, batch_size=strategy.num_replicas_in_sync)
results['BiLSTM'] = test_result
print(f"Test result: {test_result}")
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 150, 300)          23203800  
    _________________________________________________________________
    bidirectional (Bidirectional (None, 256)               439296    
    _________________________________________________________________
    dense (Dense)                (None, 1)                 257       
    =================================================================
    Total params: 23,643,353
    Trainable params: 439,553
    Non-trainable params: 23,203,800
    _________________________________________________________________
    Epoch 1/20
    202/202 [==============================] - 12s 31ms/step - loss: 0.4193 - accuracy: 0.8080 - val_loss: 0.2690 - val_accuracy: 0.8977
    
    Epoch 00001: val_loss improved from inf to 0.26902, saving model to toxiccomment_bilstm.h5
    Epoch 2/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.2534 - accuracy: 0.8971 - val_loss: 0.2386 - val_accuracy: 0.9054
    
    Epoch 00002: val_loss improved from 0.26902 to 0.23865, saving model to toxiccomment_bilstm.h5
    Epoch 3/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.2158 - accuracy: 0.9150 - val_loss: 0.2093 - val_accuracy: 0.9156
    
    Epoch 00003: val_loss improved from 0.23865 to 0.20931, saving model to toxiccomment_bilstm.h5
    Epoch 4/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1922 - accuracy: 0.9214 - val_loss: 0.2284 - val_accuracy: 0.9020
    
    Epoch 00004: val_loss did not improve from 0.20931
    Epoch 5/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1750 - accuracy: 0.9303 - val_loss: 0.2120 - val_accuracy: 0.9165
    
    Epoch 00005: val_loss did not improve from 0.20931
    Epoch 6/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1645 - accuracy: 0.9331 - val_loss: 0.2310 - val_accuracy: 0.9085
    
    Epoch 00006: val_loss did not improve from 0.20931
    
    Epoch 00006: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 7/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1583 - accuracy: 0.9374 - val_loss: 0.2004 - val_accuracy: 0.9199
    
    Epoch 00007: val_loss improved from 0.20931 to 0.20037, saving model to toxiccomment_bilstm.h5
    Epoch 8/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1382 - accuracy: 0.9471 - val_loss: 0.2006 - val_accuracy: 0.9217
    
    Epoch 00008: val_loss did not improve from 0.20037
    Epoch 9/20
    202/202 [==============================] - 5s 26ms/step - loss: 0.1292 - accuracy: 0.9498 - val_loss: 0.2019 - val_accuracy: 0.9208
    
    Epoch 00009: val_loss did not improve from 0.20037
    Epoch 10/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1281 - accuracy: 0.9506 - val_loss: 0.2040 - val_accuracy: 0.9242
    
    Epoch 00010: val_loss did not improve from 0.20037
    
    Epoch 00010: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
    Epoch 11/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1145 - accuracy: 0.9561 - val_loss: 0.2040 - val_accuracy: 0.9230
    
    Epoch 00011: val_loss did not improve from 0.20037
    Epoch 12/20
    202/202 [==============================] - 5s 23ms/step - loss: 0.1194 - accuracy: 0.9552 - val_loss: 0.2046 - val_accuracy: 0.9227
    
    Epoch 00012: val_loss did not improve from 0.20037
    Restoring model weights from the end of the best epoch.
    Epoch 00012: early stopping
    406/406 [==============================] - 7s 14ms/step - loss: 0.2261 - accuracy: 0.9196
    Test result: [0.2260790914297104, 0.9195685386657715]



    
![png](images/output_39_1.png)
    


# 6. Compare the results


```python
results = pd.DataFrame(results).transpose()
results.columns = ['loss', 'accuracy']
```


```python
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GRU</th>
      <td>0.211472</td>
      <td>0.920801</td>
    </tr>
    <tr>
      <th>LSTM</th>
      <td>0.219674</td>
      <td>0.909399</td>
    </tr>
    <tr>
      <th>BiLSTM</th>
      <td>0.226079</td>
      <td>0.919569</td>
    </tr>
  </tbody>
</table>
</div>



[<-PREV](toxiccomment.md)
