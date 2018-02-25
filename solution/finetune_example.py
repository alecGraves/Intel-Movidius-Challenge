# finetuning example for Intel Movidius using Keras
# copyright 2018 Alec Graves (shadySource)
# MIT License

seed = 13037
import numpy as np
np.random.seed(seed)
import os
from keras import optimizers, metrics, callbacks
from keras.models import Model
from keras.applications import inception_v3, inception_resnet_v2, resnet50
from keras.layers import Input, Dropout, Conv2D, Activation, Flatten, Dense, GlobalAveragePooling2D
import keras.backend as K

import datatool

batch_size = 64
input_shape = [299, 299, 3]
num_classes = 200

input_layer = Input(batch_shape=[batch_size]+input_shape)

# model = inception_resnet_v2.InceptionResNetV2(include_top=False)
base_model = inception_v3.InceptionV3(include_top=False,input_tensor=input_layer)
# base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)

x = base_model.output

# x = Flatten()(x)
# x = Dense(num_classes)(x)
# predictions = Activation('softmax')(x)
# top_model = Dense(1024)(top_model)
# top_model = Activation('relu')(top_model)
# top_model = Dropout(0.6, seed=seed)(top_model)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(200, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# view number of layers
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)
# exit()

cb = callbacks.ModelCheckpoint('./weights/keras/model.h5')

model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop')

model.fit_generator(
    datatool.train_gen(batch_size),
    validation_data=datatool.val_gen(batch_size),
    validation_steps=datatool.num_val//batch_size,
    steps_per_epoch=datatool.num_train//batch_size,
    callbacks=[cb],
    epochs=5)

for layer in base_model.layers[-40:]:
    layer.trainable = True

sgd = optimizers.SGD(lr=0.0001, momentum=0.9)
model.compile(loss=sgd,
            optimizer='rmsprop')

model.fit_generator(
    datatool.train_gen(batch_size),
    validation_data=datatool.val_gen(batch_size),
    validation_steps=datatool.num_val//batch_size,
    steps_per_epoch=datatool.num_train//batch_size,
    callbacks=[cb],
    epochs=5)