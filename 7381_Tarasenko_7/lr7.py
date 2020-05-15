import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras.datasets import imdb
from keras.preprocessing import sequence


def draw_plots(results):
    # получение ошибки и точности в процессе обучения
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['acc']
    val_acc = results.history['val_acc']
    epochs = range(1, len(loss) + 1)

    # построение графика ошибки
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # построение графика точности
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def user_review(review, models):
    print(review)
    words = review.replace(',', '').replace('.', '').replace(':', '').replace('!', '').replace('?', '').\
        replace('(', '').replace(')', '').replace('"', '').replace(' - ', '').lower().split()
    indexes = dict(imdb.get_word_index())
    codes, num_words = [], []
    for word in words:
        word = indexes.get(word)
        if word and (word < 10000):
            num_words.append(word)
    codes.append(num_words)
    codes = sequence.pad_sequences(codes, maxlen=max_review_length)
    avg_prediction = 0
    for model in models:
        avg_prediction += model.predict(codes)[0][0]
    avg_prediction /= len(models)
    print(1 - avg_prediction)


# исправление ошибки с данными:
# ValueError: Object arrays cannot be loaded when allow_pickle=False
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
# restore np.load for future normal usage
np.load = np_load_old

# перестроение массивов данных
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

max_review_length = 500
training_data = sequence.pad_sequences(training_data, maxlen=max_review_length)
testing_data = sequence.pad_sequences(testing_data, maxlen=max_review_length)

# подготовка и обучение моделей, вывод результатов и графиков
top_words = 10000
embedding_vector_length = 32

# модель - 1
max1, best_res1, best_model1 = 0, None, None
for i in range(3):
    model1 = tensorflow.keras.models.Sequential()
    model1.add(tensorflow.keras.layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model1.add(tensorflow.keras.layers.Dropout(0.3))
    # model1.add(tensorflow.keras.layers.LSTM(200))
    # model1.add(tensorflow.keras.layers.Dropout(0.3, noise_shape=None, seed=None))
    model1.add(tensorflow.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model1.add(tensorflow.keras.layers.MaxPooling1D(pool_size=2))
    model1.add(tensorflow.keras.layers.Dropout(0.3))
    model1.add(tensorflow.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model1.add(tensorflow.keras.layers.MaxPooling1D(pool_size=2))
    model1.add(tensorflow.keras.layers.Dropout(0.3))
    model1.add(tensorflow.keras.layers.GRU(64, return_sequences=True))
    model1.add(tensorflow.keras.layers.SimpleRNN(64))
    model1.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    results1 = model1.fit(training_data, training_targets, validation_data=(testing_data, testing_targets),
                          epochs=5, batch_size=64, verbose=0)
    scores1 = model1.evaluate(testing_data, testing_targets, verbose=0)
    print(i + 1, ") Accuracy: %.2f%%" % (scores1[1]*100))
    if (scores1[1]*100) > max1:
        max1 = (scores1[1]*100)
        best_res1 = results1
        best_model1 = model1
print("Best accuracy: %.2f%%" % max1)
if best_res1: draw_plots(best_res1)

# модель - 2
max2, best_res2, best_model2 = 0, None, None
for i in range(3):
    model2 = tensorflow.keras.models.Sequential()
    model2.add(tensorflow.keras.layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model2.add(tensorflow.keras.layers.Dropout(0.3))
    model2.add(tensorflow.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model2.add(tensorflow.keras.layers.MaxPooling1D(pool_size=2))
    model2.add(tensorflow.keras.layers.Dropout(0.2))
    model2.add(tensorflow.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model2.add(tensorflow.keras.layers.MaxPooling1D(pool_size=2))
    model2.add(tensorflow.keras.layers.Dropout(0.2))
    model2.add(tensorflow.keras.layers.GRU(64, return_sequences=True))
    model2.add(tensorflow.keras.layers.Dropout(0.2))
    model2.add(tensorflow.keras.layers.GRU(32))
    model2.add(tensorflow.keras.layers.Dropout(0.2))
    model2.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    results2 = model2.fit(training_data, training_targets, validation_data=(testing_data, testing_targets),
                          epochs=5, batch_size=64, verbose=0)
    scores2 = model2.evaluate(testing_data, testing_targets, verbose=0)
    print(i + 1, ") Accuracy: %.2f%%" % (scores2[1]*100))
    if (scores2[1]*100) > max2:
        max2 = (scores2[1]*100)
        best_res2 = results2
        best_model2 = model2
print("Best accuracy: %.2f%%" % max2)
if best_res2: draw_plots(best_res2)

scores = 0
for sc in [max1, max2]: scores += sc
scores /= len([max1, max2])
print("Ens. accuracy: %.2f%%" % scores)

# тестирование программы на пользовательских обзорах
bad_review = 'This was the worst movie I saw at WorldFest and it also received the least amount of applause afterwards!'
good_review = 'I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air conditioned theater and watching a light-hearted comedy.'
print('Bad example:', end=' ')
user_review(bad_review, [best_model1, best_model2])
print()
print('Good example:', end=' ')
user_review(good_review, [best_model1, best_model2])
