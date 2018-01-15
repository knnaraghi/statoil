import numpy as np
random_seed = np.random.seed(42)

from keras.preprocessing.image import ImageDataGenerator

augment = ImageDataGenerator(width_shift_range=0.0,
                            height_shift_range=0.0,
                            horizontal_flip=True,
                            vertical_flip=False, 
                            zoom_range=0.2, 
                            rotation_range=10,
                            shear_range=0.0)

def augment_generator(X, X_angle, y, batch_size=32):
    gen_X = augment.flow(X, y, batch_size=batch_size, seed=random_seed)
    gen_angle = augment.flow(X, X_angle, batch_size=batch_size, seed=random_seed)
    
    while True:
        X_i = gen_X.next()
        X_angle_i = gen_angle.next()
        
        yield [X_i[0], X_angle_i[1]] , X_i[1]