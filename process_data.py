import numpy as np
import pandas as pd
import cv2

def processData(train, test, model):
    
    train = pd.read_json(train)
    test = pd.read_json(test)

    #sanity checks
    print(train.shape)
    print(test.shape)
    
    test.head()
    
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    
    X_train_angle = np.array(train.inc_angle)
    
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    
    if model=='InceptionV3':
        x_band1 = np.array([cv2.resize(band, (139, 139)).astype(np.float32) for band in x_band1])
        x_band2 = np.array([cv2.resize(band, (139, 139)).astype(np.float32) for band in x_band2])
        
    if model=='Resnet50':
        x_band1 = np.array([cv2.resize(band, (200, 200)).astype(np.float32) for band in x_band1])
        x_band2 = np.array([cv2.resize(band, (200, 200)).astype(np.float32) for band in x_band2])
    
    X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],
                         ((x_band1 + x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
    
    x_test_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    x_test_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    
    test.inc_angle = test.inc_angle.replace('na', 0)
    test.inc_angle = test.inc_angle.astype(float).fillna(0.0)
    
    X_test_angle = np.array(test.inc_angle)
    
    if model=='Resnet50':
        x_test_band1 = np.array([cv2.resize(band, (200, 200)).astype(np.float32) for band in x_test_band1])
        x_test_band2 = np.array([cv2.resize(band, (200, 200)).astype(np.float32) for band in x_test_band2])
        
    X_test = np.concatenate([x_test_band1[:, :, :, np.newaxis], x_test_band2[:, :, :, np.newaxis],
                         ((x_test_band1 + x_test_band2)/2)[:, :, :, np.newaxis]], axis=-1)
    
    y_train = np.array(train["is_iceberg"])
    
    return X_train, X_train_angle, X_test, X_test_angle, y_train