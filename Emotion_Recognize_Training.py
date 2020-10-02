import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
import matplotlib.pyplot as plt

df=pd.read_csv('./fer2013/fer2013.csv')

X_train,train_y,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")


num_features = 64
num_labels = 7
batch_size = 64
epochs = 1
width, height = 48, 48

from sklearn.model_selection import train_test_split
X_test, X_validate, test_y, y_validate = train_test_split(X_test,test_y,test_size=0.5,random_state=0)

X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')
X_validate= np.array(X_validate,'float32')
y_validate = np.array(y_validate,'float32')


train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)
y_validate = np_utils.to_categorical(y_validate, num_classes = num_labels)


X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)
X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)
X_validate-=np.mean(X_validate, axis=0)
X_validate/=np.std(X_validate, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
X_validate = X_validate.reshape(X_validate.shape[0],48,48,1)

def create_model():
    """
    Create deep learning model for emotion detection.
    """
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation='softmax'))

model = create_model()
model.summary()

model.compile(loss=categorical_crossentropy,
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)

fer1_json = model.to_json()
with open("fer1.json", "w") as json_file:
    json_file.write(fer1_json)
model.save_weights("fer1.h5")
