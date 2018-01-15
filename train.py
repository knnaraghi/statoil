import numpy as np
import pandas as pd
random_seed = np.random.seed(42)

from process_data import processData as processData
from data_generator import augment_generator as augment_generator
from transfer_model import transferModel as transferModel

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

### Input Model Name ### 
### See process_data.py for different models ###
model = 'vgg16'
X_train, X_train_angle, X_test, X_test_angle, y_train = processData("train.json", "test.json", model)

def get_callbacks(filepath):
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0001, 
                          patience=7, verbose=1, mode='min')
    checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, 
                              epsilon=1e-4, mode='min')
    
    return [earlyStop, checkpointer, reduce_lr]

def transferCV(X_train, X_train_angle, X_test, X_test_angle, y_train, model, finetune=False): 
    K = 3
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, 
                                 random_state=random_seed).split(X_train, y_train))
    y_test_pred_log = 0
    y_train_pred_log = 0
    y_valid_pred_log = 0.0*y_train
    
    for idx, (train_idx, val_idx) in enumerate(folds):
        print("Training on Fold\n", idx)
        
        X_train_idx, X_val_idx = X_train[train_idx], X_train[val_idx]
        y_train_idx, y_val_idx = y_train[train_idx], y_train[val_idx]
        
        X_train_angle_idx, X_val_angle_idx = X_train_angle[train_idx], X_train_angle[val_idx]
        
        file_path = "%s_cv_weights.hdf5"%idx
        
        callbacks_list = get_callbacks(filepath=file_path)
        
        augment_flow = augment_generator(X_train_idx, X_train_angle_idx, y_train_idx)
        
        cvModel = transferModel(model)
        
        ###Finetune has to be changed for each respective model architecture
        
        if finetune:
            cvModel.load_weights(filepath=file_path)
            
            for layer in cvModel.layers[15:]:
                layer.trainable = True
                cvModel.compile(optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), 
                        loss='binary_crossentropy', metrics=['accuracy'])
        
        cvModel.fit_generator(augment_flow, epochs=100, steps_per_epoch=len(X_train_idx) / 32, 
                              validation_data=([X_val_idx, X_val_angle_idx], y_val_idx),
                             callbacks=callbacks_list)
        
        cvModel.load_weights(filepath=file_path)
        
        score = cvModel.evaluate([X_train_idx, X_train_angle_idx], y_train_idx, verbose=0)
        
        #Train Score
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        
        #Validation Score
        score = cvModel.evaluate([X_val_idx, X_val_angle_idx], y_val_idx, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        predictions_valid = cvModel.predict([X_val_idx, X_val_angle_idx])
        y_valid_pred_log[val_idx] = predictions_valid.reshape(predictions_valid.shape[0])
        
        temp_test = cvModel.predict([X_test, X_test_angle])
        y_test_pred_log += temp_test.reshape(temp_test.shape[0])
        
        temp_train = cvModel.predict([X_train, X_train_angle])
        y_train_pred_log += temp_train.reshape(temp_train.shape[0])
        
    y_test_pred_log = y_test_pred_log / K
    y_train_pred_log = y_train_pred_log / K
    
    print('Train Log Loss Validation= ', log_loss(y_train, y_train_pred_log))
    print('Valid Log Loss Validation= ', log_loss(y_train, y_valid_pred_log))
    
    return y_test_pred_log

predictions = transferCV(X_train, X_train_angle, X_test, X_test_angle, y_train, model='vgg16', finetune=False)

test = pd.read_json("test.json")

predictions_df = test[['id']].copy()
predictions_df['is_iceberg'] = predictions

predictions_df.to_csv('submission.csv', index=False)
