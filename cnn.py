import tensorflow as tf

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

train_path = "dataset/train"
valid_path = "dataset/valid"
test_path = "dataset/test"

image_shape = (350,350,3)
N_CLASSES = 2
BATCH_SIZE = 256

train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
train_generator = train_datagen.flow_from_directory(train_path, batch_size = BATCH_SIZE, target_size = (350,350), class_mode = 'categorical')

valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
valid_generator = valid_datagen.flow_from_directory(valid_path, batch_size = BATCH_SIZE, target_size = (350,350), class_mode = 'categorical')

test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
test_generator = test_datagen.flow_from_directory(test_path, batch_size = BATCH_SIZE, target_size = (350,350), class_mode = 'categorical')

weight_decay = 1e-3

model = Sequential([
    Conv2D(filters = 8 , kernel_size = 2, activation = 'relu', input_shape = image_shape),
    MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 16 , kernel_size = 2, activation = 'relu', input_shape = image_shape),
    MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 32 , kernel_size = 2, activation = 'relu', kernel_regularizer = regularizers.l2(weight_decay)),
    MaxPooling2D(pool_size = 2),
    
    Dropout(0.4),
    Flatten(),
    Dense(300,activation='relu'),
    Dropout(0.5),
    Dense(2,activation='softmax')
])

model.summary()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

checkpointer = ModelCheckpoint('first_model.hdf5',verbose=1, save_best_only= True)
early_stopping = EarlyStopping(monitor= 'val_loss', patience= 10)
optimizer = optimizers.Adam(learning_rate= 0.00001, decay= 1e-5)
model.compile(loss= 'categorical_crossentropy', optimizer= optimizer, metrics=['AUC','acc'])

history = model.fit(train_generator, epochs = 50, verbose = 1, validation_data = valid_generator, callbacks = [checkpointer, early_stopping])

plt.plot(history.history['acc'], label = 'train',)
plt.plot(history.history['val_acc'], label = 'valid')
plt.legend(loc = 'lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

result = model.evaluate(test_generator)