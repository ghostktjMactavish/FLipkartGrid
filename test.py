from keras.layers import Dense,Conv2D,Input,Reshape
import keras.layers as layers
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.optimizers import Adam
import keras
import numpy

def data_generate():
    image = []
    co_ordinates = []

    while 1:
        fhandle = open('training.csv','r')
        num_images=0
        for line in fhandle:
            num_images = num_images + 1
            line_tok = line.split(',')
            img_name = 'training_images/'+line_tok[0]
            img = load_img(img_name,target_size=(160,120))
            img = img_to_array(img)
            img = img/255
            points = [int(line_tok[1])/640,int(line_tok[2])/640,int(line_tok[3])/480,int(line_tok[4])/480]
            image.append(img)
            co_ordinates.append(points)
            if num_images%50 == 0:
                image = numpy.array(image)
                co_ordinates = numpy.array(co_ordinates)
                yield [image, co_ordinates]
                image = []
                co_ordinates = []
        fhandle.close()

optimize=Adam(lr=0.00001)
input_image = Input(shape=(160,120,3))

conv_layer1 = layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same')(input_image)
conv_layer1 = layers.BatchNormalization()(conv_layer1)
conv_layer1 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer1)

pool_layer1=layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_layer1)

conv_layer2 = layers.Conv2D(32, (3, 3), strides=(1, 1),padding='same')(pool_layer1)
conv_layer2 = layers.BatchNormalization()(conv_layer2)
conv_layer2 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer2)

pool_layer2=layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_layer2)

conv_layer3 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(pool_layer2)
conv_layer3 = layers.BatchNormalization()(conv_layer3)
conv_layer3 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer3)

#pool_layer1=layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv_layer3)

conv_layer4 = layers.Conv2D(64, (3, 3), strides=(1, 1),padding='same')(conv_layer3)
conv_layer4 = layers.BatchNormalization()(conv_layer4)
conv_layer4 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer4)

conv_layer5 = layers.Conv2D(32, (1, 1), strides=(1, 1),padding='same')(conv_layer4)
conv_layer5 = layers.BatchNormalization()(conv_layer5)
conv_layer5 = layers.advanced_activations.LeakyReLU(alpha=0.1)(conv_layer5)

pool_layer3=layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_layer5)

flat_layer = layers.Flatten()(pool_layer3)
#dense_layer1 = layers.Dense(4096,activation='relu')(flat_layer)
#dense_layer2 = layers.Dense(4096,activation='relu')(dense_layer1)
dropout=layers.Dropout(0.2)(flat_layer)
dense_layer = layers.Dense(1024,activation='sigmoid')(dropout)
#dropout1 = layers.Dropout(0.5)(dense_layer)
output_layer = layers.Dense(4,activation='sigmoid')(dense_layer)

Box_model=Model(inputs=input_image, outputs=output_layer)
Box_model.compile(loss='mse', optimizer=optimize, metrics=['accuracy'])

Box_model.load_weights('my_model_weights1.h5')

output_file = open('result.csv','w')
test_file = open('test.csv','r')
counter = 0
for line in test_file:
    counter = counter + 1
    if counter%200 == 0:
        print(counter)
    img_name = line.split(',')[0]
    img_name = "test_images/"+img_name
    img = load_img(img_name,target_size=(160,120))
    img = img_to_array(img)
    img = [img]
    img = numpy.array(img)
    Box = Box_model.predict(img)
    Box = Box[0]
    data_to_write = line.split(',')[0]+","+str(int(Box[0]*640))+","+str(int(Box[1]*640))+","+str(int(Box[2]*480))+","+str(int(Box[3]*480))+"\n"
    output_file.write(data_to_write)

output_file.close()
test_file.close()
