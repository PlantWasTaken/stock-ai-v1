import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from tensorflow.keras import layers
from tensorflow.compat.v2.data import Dataset

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices())

SEQ_LEN = 15
BATCH_SIZE = 16

df = pd.read_csv(r'E:\ai\stock\stock\v2.0\test.csv')


close = df['Close'].values.tolist()[SEQ_LEN:] #removing first SEQ_LEN indexes due to shift in data
del df['Close']


df_list = df.values.tolist()


feat = np.array([df_list[i:SEQ_LEN+i] for i in range(len(df_list)-SEQ_LEN)])
lab = np.array(close) 

validation_samples = 100
feat_val = feat[:-validation_samples]
lab_val = lab[:-validation_samples]
feat = feat[-validation_samples:]
lab  = lab [-validation_samples:]

print(len(feat_val),len(feat))

exit()
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) #distrbuted learning
def generate_data():
    while True:
        #try:
            random_val = random.randint(0,len(labels)-seq_len)
            x = np.array([features_train[random_val+i] for i in range(batch_size)]) #labels
            y = np.array([labels_train[random_val+i] for i in range(batch_size)]) #features
            print(x.shape,y.shape)
            #yield x,y
        #except IndexError:
        #    pass
generate_data()
exit()

def model_fn():
    with strategy.scope():
        size = 256
        inputs = keras.Input(shape=(seq_len,row))
        rnn = keras.layers.LSTM(size, return_sequences=True)(inputs)
        rnn = keras.layers.Dropout(0.5)(rnn)
        for _ in range(9):
            rnn = keras.layers.LSTM(size, return_sequences=True)(rnn)
            rnn = keras.layers.Dropout(0.5)(rnn)

        dense = keras.layers.Dense(size, activation='relu')(inputs)
        dense = keras.layers.Dropout(0.2)(dense)
        for _ in range(9):
            dense = keras.layers.Dense(size, activation='relu')(dense)
            dense = keras.layers.Dropout(0.2)(dense)

        conc = keras.layers.concatenate([rnn,dense])
        conc = keras.layers.Dense(64,activation='relu')(conc)
        conc = keras.layers.Dense(32,activation='relu')(conc)


        outputs = layers.Dense(1,activation='linear')(conc)
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(lr=0.0001),
        #metrics = ['accuracy'],
    )
    return model



model = model_fn()
#with strategy.scope():
#    model = tf.keras.models.load_model(r'C:\Users\Max\Desktop\ai\stock\Stock015.h5')
distributed_model = model
model.summary()
    

output_types = (tf.float32, tf.float32)
outpus_shapes= ([seq_len,5],[seq_len,1])

train_data = Dataset.from_generator(generate_data,output_types=output_types) #fixes bugs / memory sharidng
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
train_data = train_data.with_options(options)

distributed_model.fit(train_data, steps_per_epoch = ((len(df)-seq_len )// (batch_size)), epochs = 2000)
distributed_model.evaluate(features_eval, labels_eval, batch_size=128)

distributed_model.save('Stock015.h5',save_format="h5")