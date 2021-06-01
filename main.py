import keras.activations
import tensorflow as tf
import numpy as np
import struct as st
import gc

'''Pierwsze nieudane podejście
(x_train, y_train), (x_test, y_test) = mnist.load_data()

a = open("Pol_EMNIST/1000/pol_EMNIST_letters_images_labels1000", "rb")
b = a.read()
a.close()
print(f'num of labels {int.from_bytes(b[4:8],"big")}')

a = open("Pol_EMNIST/1000/pol_EMNIST_letters_images1000", "rb")
c = a.read()
a.close()

print(f'num of images {int.from_bytes(c[4:8],"big")}\n{int.from_bytes(c[8:12],"big")}x{int.from_bytes(c[12:16],"big")}')
label = []
images = []

for i in range(1, 1000):
    label.append(b[8+i])
    img = []
    for j in range(28*28):
        img.append(c[(16*i)+j])
    img = np.reshape(np.array(img), (28, 28))
    images.append(img)

a = open("train-labels-idx1-ubyte", "rb")
b = a.read()
a.close()

a = open("train-images-idx3-ubyte", "rb")
c = a.read()
a.close()

for i in range(1, 10000):
    label.append(b[8+i])
    img = []
    for j in range(28*28):
        img.append(c[(16*i)+j])
    img = np.reshape(np.array(img), (28, 28))
    images.append(img)

test_x = []

test_y = []

for i in range(10000, 2*10000):
    test_y.append(b[8+i])
    img = []
    for j in range(28*28):
        img.append(c[(16*i)+j])
    img = np.reshape(np.array(img),(28,28))
    test_x.append(img)

train_yOHE = to_categorical(label,num_classes = 26,dtype='int')
test_yOHE = to_categorical(test_y,num_classes = 26,dtype='int')


images = tf.keras.utils.normalize(images,axis=1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)
images = images.reshape(images.shape[0],28,28,1)
test_x = test_x.reshape(test_x.shape[0],28,28,1)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(26,activation ="softmax"))
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(images.shape)
print(np.array(label).shape)

history = model.fit(images,train_yOHE,batch_size=32,validation_split=0.1,epochs=10)
'''


def file_to_image_array(file):
    file.seek(0)
    magic = st.unpack('>4B', file.read(4))
    number_of_images = st.unpack('>I', file.read(4))[0]
    number_of_rows = st.unpack('>I', file.read(4))[0]
    number_of_columns = st.unpack('>I', file.read(4))[0]
    n_bytes_total = number_of_images * number_of_rows * number_of_columns * 1
    images_array = np.asarray(st.unpack('>' + 'B' * n_bytes_total, file.read(n_bytes_total))) \
        .reshape((number_of_images, number_of_rows, number_of_columns))
    return images_array


def file_to_label_array(file):
    file.seek(0)
    magic = st.unpack('>4B', file.read(4))
    number_of_labels = st.unpack('>I', file.read(4))[0]
    n_bytes_total = number_of_labels * 1
    labels_array = np.asarray(st.unpack('>' + 'B' * n_bytes_total, file.read(n_bytes_total))) \
        .reshape(number_of_labels)
    return labels_array, number_of_labels


'------------------------------------------ Bazowy mnist zawiera tylko liczby 0-9--------------------------------------'
'''
mnist = tf.keras.datasets.mnist
(digits_data_train, digits_labels_train), (digits_data_test, digits_labels_test) = mnist.load_data()
digits_data_train, digits_data_test = digits_data_train / 255.0, digits_data_test / 255.0
#oddaj ram złodzieju
del digits_labels_train, digits_labels_test, digits_data_test, digits_data_train
gc.collect()
'''
'------------------------------------------ EMNIST znaków ogólnych-----------------------------------------------------'
file_train_images = open("datasets\emnist_train_images", "rb")
emnist_data_train = file_to_image_array(file_train_images) / 255.0
file_train_images.close()
file_train_labels = open("datasets\emnist_train_labels", "rb")
emnist_labels_train, emnist_number_of_labels_train = file_to_label_array(file_train_labels)
file_train_labels.close()
file_test_images = open("datasets\emnist_test_images", "rb")
emnist_data_test = file_to_image_array(file_test_images) / 255.0
file_test_images.close()
file_test_labels = open("datasets\emnist_test_labels", "rb")
emnist_labels_test, emnist_number_of_labels_test = file_to_label_array(file_test_labels)
file_test_labels.close()

#mapa labeli jest w numerach, trzeba konwertować do ascii

'------------------------------------------ EMNIST znaków polskich Ą i Ć-----------------------------------------------'
file_train_images_pol = open("datasets\emnist_pol_letters_images", "rb")
emnist_pol_data = file_to_image_array(file_train_images_pol) / 255.0
file_train_images_pol.close()
file_train_labels_pol = open("datasets\emnist_pol_letters_images_labels", "rb")
emnist_pol_labels, emnist_pol_number_of_labels = file_to_label_array(file_train_labels_pol)
file_train_labels_pol.close()
#podział na train i test
emnist_pol_labels = emnist_pol_labels + 35
print(emnist_pol_labels[0:100])
emnist_pol_data_test = emnist_pol_data[800:len(emnist_pol_data)-1]
emnist_pol_data = emnist_pol_data[0:799]
emnist_pol_labels_test = emnist_pol_labels[800:len(emnist_pol_labels)-1]
emnist_pol_labels = emnist_pol_labels[0:799]

'------------------------------------------ Zmienne do mielenia, pamiętać o liczbie labeli ----------------------------'
#number_of_unique_labels = 10           #ilość unikatowych labeli w mnist
number_of_unique_labels = max(len(np.unique(emnist_labels_train)), len(np.unique(emnist_labels_test))) \
                          + len(np.unique(emnist_pol_labels))
print(number_of_unique_labels)
#number_of_unique_labels = emnist_pol_number_of_labels
epoch_count = 10                                                                    #10 -> 20 -> 10 po 12 accuracy spada

#input_data, input_labels = digits_data_train, digits_labels_train
#test_data, test_labels = digits_data_test, digits_labels_test

#input_data, input_labels = emnist_data_train, emnist_labels_train
#test_data, test_labels = emnist_data_test, emnist_labels_test

input_data = np.concatenate((emnist_data_train, emnist_pol_data), axis=0)
input_labels = np.concatenate((emnist_labels_train, emnist_pol_labels), axis=0)
test_data = np.concatenate((emnist_data_test, emnist_pol_data_test), axis=0)
test_labels = np.concatenate((emnist_labels_test, emnist_pol_labels_test), axis=0)
'''
input_data = emnist_pol_data
input_labels = emnist_pol_labels
test_data = emnist_pol_data_test
test_labels = emnist_pol_labels_test
'''
load_model = False
'------------------------------------------ Strefa modelu, tu nie ruszać-----------------------------------------------'
if load_model:
    model = keras.models.load_model("save")
else:
    model = tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28)),
         tf.keras.layers.Dense(192, activation='relu'),  #128 -> 256 -> 512 -> 128 -> 192
         #tf.keras.layers.Dropout(0.1)                   #0.2 -> 0.1 -> 0.05 -> 0.1
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(256, activation='relu'),
         tf.keras.layers.Dense(number_of_unique_labels)])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(input_data, input_labels, epochs=epoch_count)
model.evaluate(test_data, test_labels, verbose=2)
model.evaluate(emnist_pol_data_test, emnist_pol_labels_test, verbose=2)
model.save("save")
