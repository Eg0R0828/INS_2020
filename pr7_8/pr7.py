import numpy as np
import tensorflow
import random
import math
import matplotlib.pyplot as plt


def func(i):
    return ((i % 20) + 1) / 20


def gen_sequence(seq_len=1000):
    seq = [abs(math.sin(i/20)) + func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)


def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)), seq)
    plt.show()


def gen_data_from_sequence(seq_len=1006, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return (past, future)


data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(tensorflow.keras.layers.LSTM(32, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
model.add(tensorflow.keras.layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(tensorflow.keras.layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_data, train_res, epochs=50, validation_data=(val_data, val_res), verbose=0)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss)
plt.plot(range(len(val_loss)), val_loss)
plt.title('LOSS')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length, predicted_res)
plt.plot(pred_length, test_res)
plt.title('SEQUENCE')
plt.ylabel('Sequence')
plt.xlabel('Argument')
plt.show()
