from keras.layers import GlobalMaxPooling2D, Dropout, Dense, Input
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16, VGG19
from keras.applications.xception import Xception 
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50


def transferModel(transfer_model):
    ang_input = Input(shape=(1,))
    angle_layer = Dense(1,)(ang_input)
    
    if transfer_model == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(75, 75, 3), classes=1)
    
    if transfer_model == 'vgg19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(75, 75, 3), classes=1)
    
    if transfer_model == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(75, 75, 3), classes=1)
        
    if transfer_model == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(139, 139, 3), classes=1)
        
    if transfer_model == 'Resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3), classes=1)
        
    for layer in base_model.layers:
        layer.trainble = False
        
    transferModel = base_model.output
    
    transferModel = GlobalMaxPooling2D()(transferModel)
    transferModel = concatenate([transferModel, angle_layer])
    transferModel = Dense(512, activation='relu', name='fc2')(transferModel)
    transferModel = Dropout(0.3)(transferModel)
    transferModel = Dense(512, activation='relu', name='fc3')(transferModel)
    transferModel = Dropout(0.3)(transferModel)
    
    predictions = Dense(1, activation='sigmoid')(transferModel)
    
    model = Model([base_model.input, ang_input], output=predictions)
    
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model