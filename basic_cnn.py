from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

def cnn_model():
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size = (3,3), activation='relu', input_shape=(75,75,3)))
    model.add(Conv2D(64, kernel_size = (3,3), activation='relu'))
    model.add(Conv2D(64, kernel_size = (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(128, kernel_size = (3,3), activation='relu'))
    model.add(Conv2D(128, kernel_size = (3,3), activation='relu'))
    model.add(Conv2D(128, kernel_size = (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(128, kernel_size = (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(256, kernel_size = (3,3), activation='relu', name='output'))
    
    return model