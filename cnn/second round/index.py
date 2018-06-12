from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

model = Sequential([
    # Conv layer
    Conv2D(32,(3,3), input_shape=(150,150,3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64,(3,3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128,(3,3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),

    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),

    # Dense layer
    Flatten(),
    Dense(128),
    Activation('relu'),
    # Dropout(0),
    Dense(1),
    Activation('sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

batch_size = 16

# this is the augmentation configuration we will use fro training
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    '../data/train', # this is the target directory
    target_size=(150,150), # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='binary', # since we use binary_crossentropy loss, we need binary labels
    shuffle=False
)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

model.fit_generator(
    train_generator,
    steps_per_epoch=20000 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size
)
model.save_weights('first_try.h5')  # always save your weights after training or during training
model.save('model.h5')