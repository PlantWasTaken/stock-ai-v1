import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras import layers

#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#print(tf.config.list_physical_devices())

ROW = 12
SEQ_LEN = 15
BATCH_SIZE = 16

df = pd.read_csv(r'E:\ai\stock\stock\v2.0\test.csv')


close = df['Close'].values.tolist()[SEQ_LEN:] #removing first SEQ_LEN indexes due to shift in data
del df['Close']


df_list = df.values.tolist()


feat = np.array([df_list[i:SEQ_LEN+i] for i in range(len(df_list)-SEQ_LEN)])
lab = np.array(close) 

#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) #distrbuted learning
def generate_data():
    while True:
            BATCH_SIZE = 1
            random_val = random.randint(0,len(lab)-SEQ_LEN)
            x = np.array(feat[random_val:random_val+BATCH_SIZE]) #labels
            y = np.array(lab[random_val:random_val+BATCH_SIZE] ) #features

            x = np.expand_dims(x,axis=-1)
            yield x,y

#generate_data()


def model_fn():

    input_layer = layers.Input(shape=(SEQ_LEN,ROW,))
    
    lstm_1 = layers.LSTM(64, return_sequences=True)(input_layer)
    lstm_2 = layers.LSTM(64, return_sequences=False)(lstm_1)
    
    output_layer = layers.Dense(1, name="layer3")(lstm_2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
    )

    return model

    

model = model_fn()

model.summary()
    

model.fit(feat, lab, batch_size=64, epochs=2, validation_split=0.2)

#distributed_model.save('Stock015.h5',save_format="h5")

