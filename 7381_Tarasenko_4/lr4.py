import tensorflow, numpy, PIL.Image
import matplotlib.pyplot as plt


def user_image_test(image_file_path):
    image_file_path = './' + image_file_path
    image = numpy.array(PIL.Image.open(image_file_path).convert('L').resize((28, 28)))
    # белым по черному или черным по белому?))) + нормализация
    image = (255 - image) / 255
    return numpy.expand_dims(image, axis=0)


def testing(curr_model, image):
    print(numpy.argmax(curr_model.predict_on_batch(image)))


def create_model():
    # загрузка тренировочных и проверочных данных
    mnist = tensorflow.keras.datasets.mnist
    (train_images, train_labels),(test_images, test_labels) = mnist.load_data()

    # проверка корректности загрузки
    plt.imshow(train_images[0], cmap=plt.cm.binary)
    plt.show()
    print(train_labels[0])

    # нормализация входных данных
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # перевод правильных ответов в категориальные вектора
    train_labels = tensorflow.keras.utils.to_categorical(train_labels)
    test_labels = tensorflow.keras.utils.to_categorical(test_labels)

    # архитектура сети
    model = tensorflow.keras.models.Sequential()

    model.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # обучение сети
    H = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels),
                  verbose=0)

    # получение ошибки и точности в процессе обучения
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['acc']
    val_acc = H.history['val_acc']
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

    # проверка распознавания контрольного набора
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_loss:', test_loss)
    print('test_acc:', test_acc)

    return model


model = create_model()
while True:
    print('Educate (Press ENTER) / Testing (print an image file path) OR print `q` for quit?')
    req = input()
    if req == 'q': break
    elif req == '': model = create_model()
    else: testing(model, user_image_test(req))
