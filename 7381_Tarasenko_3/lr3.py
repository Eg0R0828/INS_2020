import matplotlib.pyplot as plt
import numpy as np
import tensorflow


def build_model():
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(tensorflow.keras.layers.Dense(64, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def plots():
    epochs = range(1, len(loss[len(loss) - 1]) + 1)

    # построение графика ошибки
    plt.plot(epochs, loss[len(loss) - 1], 'bo', label='Training loss')
    plt.plot(epochs, val_loss[len(loss) - 1], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # построение графика точности
    plt.clf()
    plt.plot(epochs, mae[len(loss) - 1], 'bo', label='Training mae')
    plt.plot(epochs, val_mae[len(loss) - 1], 'b', label='Validation mae')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


def add(x, y):
    for i in range(len(x)):
        x[i] += y[i]


# подготовка рабочих данных
(train_data, train_targets), (test_data, test_targets) = tensorflow.keras.datasets.boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

print(test_targets)

# нормализация данных
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# статистические данные для построения графиков
loss = []
val_loss = []
mae = []
val_mae = []
avg_loss = []
avg_val_loss = []
avg_mae = []
avg_val_mae = []

# перекрестная проверка по К блокам
k = 8
num_val_samples = len(train_data) // k
num_epochs = 50
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                  validation_data=(val_data, val_targets))

    # получение ошибки и точности в процессе обучения, построение графиков
    loss.append(H.history['loss'])
    val_loss.append(H.history['val_loss'])
    mae.append(H.history['mean_absolute_error'])
    val_mae.append(H.history['val_mean_absolute_error'])
    if len(avg_loss) == 0:
        avg_loss += H.history['loss']
        avg_val_loss += H.history['val_loss']
        avg_mae += H.history['mean_absolute_error']
        avg_val_mae += H.history['val_mean_absolute_error']
    else:
        add(avg_loss, H.history['loss'])
        add(avg_val_loss, H.history['val_loss'])
        add(avg_mae, H.history['mean_absolute_error'])
        add(avg_val_mae, H.history['val_mean_absolute_error'])
    #plots()

    # результат работы текущей модели
    val_mse, val_MAE = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_MAE)

# вывод усредненных графиков и ответа
for i in range(len(avg_loss)):
    avg_loss[i] /= k
    avg_val_loss[i] /= k
    avg_mae[i] /= k
    avg_val_mae[i] /= k
loss.append(avg_loss)
val_loss.append(avg_val_loss)
mae.append(avg_mae)
val_mae.append(avg_val_mae)
plots()
print(np.mean(all_scores))
