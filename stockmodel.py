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

ROW = 7
SEQ_LEN = 15
BATCH_SIZE = 16

df = pd.read_csv('test.csv')


close = df['Close'].values.tolist()[SEQ_LEN:] #removing first SEQ_LEN indexes due to shift in data
del df['Close']

df_list = df.values.tolist()

feat = np.array([df_list[i:SEQ_LEN+i] for i in range(len(df_list)-SEQ_LEN)])
lab = np.array(close) 

val = 100
feat_val = feat[-val:]
lab_val = lab[-val:]
feat = feat[:-val]
lab = lab[:-val]
def model_fn():

    input_layer = layers.Input(shape=(SEQ_LEN,ROW,))
    
    x = layers.LSTM(128, return_sequences=True)(input_layer)
    x = layers.Dropout(0.25)(x)

    for _ in range(3):
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.25)(x)

    x = layers.LSTM(64, return_sequences=False)(x)
    
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64)(x)

    output_layer = layers.Dense(1, name="layer3")(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.Accuracy()]

    )

    return model

    

model = model_fn()

model.summary()
    

model.fit(feat, lab, batch_size=64, epochs=200)
model.evaluate(feat_val,lab_val)
#prediction

ft = feat[0]
ft = np.expand_dims(ft,axis=0)
val = model.predict(ft)[0][0] #2321.0 answer
print("Prediction: " + str(val))
print("Prediction is off by: "+str(((2321-val)/2321)*100)+"%")
model.save('stonk200.h5',save_format="h5")

