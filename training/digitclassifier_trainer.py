import argparse
import os
import numpy as np
import cv2
from models import Sudokunet
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255.0
    return img

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model after training")
args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 10
BS = 128
best_weight = ModelCheckpoint(args['model'], monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, min_lr=1e-6)
CALLBACKS = [best_weight, reduce_lr]

print("[INFO] accessing data...")
data = os.listdir("Digits")
data_X, data_Y = [], []
num_classes = len(data)
for i in range(num_classes):
    data_list = os.listdir(f"Digits/{i}")
    for j in data_list:
        img = cv2.imread(f"Digits/{i}/{j}")
        img = cv2.resize(img, (32, 32))
        data_X.append(img)
        data_Y.append(i)
if len(data_X) == len(data_Y):
    print(f"Total Datapoints Extracted: {len(data_X)}")
else: print("Not all data extracted")
data_X = np.array(data_X)
data_Y = np.array(data_Y)

# Shuffling and Splitting Data
train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, test_size=0.05, random_state=42)
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# Preprocessing Data Extracted
train_X = np.array(list(map(preprocess, train_X)))
test_X = np.array(list(map(preprocess, test_X)))
valid_X = np.array(list(map(preprocess, valid_X)))
train_X = np.expand_dims(train_X, axis=-1)
test_X = np.expand_dims(test_X, axis=-1)
valid_X = np.expand_dims(valid_X, axis=-1)
le = LabelBinarizer()
train_y = le.fit_transform(train_y)
test_y = le.transform(test_y)
valid_y = le.transform(valid_y)

# Augmentation of Data
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(train_X)

# Compiling Model
model = Sudokunet.SudokuNet.build(32, 32, 1, 10)
opt = RMSprop(lr=0.001, rho=0.9, epsilon = 1e-08, decay=0.0)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

# Training Model
history = model.fit(datagen.flow(train_X, train_y, batch_size=32), epochs = 30, validation_data = (valid_X, valid_y), verbose = 1, steps_per_epoch= 200, callbacks=CALLBACKS)

# Evaluating Model
print("[INFO] evaluating network...")
predictions = model.predict(test_X)
print(classification_report(
	test_y.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))
