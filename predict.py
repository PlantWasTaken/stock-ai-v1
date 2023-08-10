import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras import layers
from keras.models import load_model
from tqdm import tqdm

SEQ_LEN = 15

model = load_model('stonk200.h5')
df = pd.read_csv('test.csv')

close = df['Close'].values.tolist()[SEQ_LEN:] #removing first SEQ_LEN indexes due to shift in data
#del df['Close']

df15 = df[0:SEQ_LEN]

def format(df15):
    mov = []
    val = df15[SEQ_LEN-1] - df15[SEQ_LEN-2]
    if(val < 0):
        mov.append(-1)
    else:
        mov.append(1)


    def sma(v):
        mva = (sum(list(df15[SEQ_LEN-v:SEQ_LEN]))/v)
        #print(mva)
        return float(mva)

    def wma(v):
        val = list(df15[SEQ_LEN-v:SEQ_LEN  ])
        val = sum([((val[j]*(v-j))) for j in range(len(val))])/sum(h for h in range(v+1))
        return float(val)

    lst = [df15[SEQ_LEN-1], mov[0], sma(5),sma(8),sma(13),wma(5),wma(8),wma(13)]
    
    new_row = ['Close','mov','5sMvA','8sMvA','13sMvA','5wMvA','8wMvA','13wMvA']

    row = {new_row[i]:lst[i] for i in range(len(lst))}
    #print(row)
    return row,lst

def df_to_np(val):
    global df15

    lst = df15['Close'].values.tolist()[1:]
    lst.append(val)
    row,lst = format(lst)

    #df15 = df15.append(row, ignore_index=True)
    df15 = pd.concat([df15, pd.DataFrame([row])], ignore_index=True)
    df15 = df15.tail(-1)

    tmp = df15.copy()
    del tmp['Close']
    tmp = np.array(tmp.values.tolist()[0:SEQ_LEN-1])
    tmp = np.expand_dims(tmp,axis=0)
    return tmp

#print(df15)
#df_to_np(99999)
#print(df15)

#formatting data for first prediction
predictions = []
prd = df.copy()
del prd['Close']
prd = np.array(prd.values.tolist()[0:SEQ_LEN])
prd = np.expand_dims(prd,axis=0)

#prediction loop
for i in range(10):
    pred = model.predict(prd)
    print(pred)
    predictions.append(pred[0][0])
    prd = df_to_np(pred)
    #print(prd)

print(len(predictions))
print(len(close[0:10]))
print(predictions)
print(close[0:10])

Y1 = predictions
Y2 = close[0:10]

import plotly.express as px
fig = px.line(y=[Y1,Y2], title='Line Graph for Two Lines')
fig.show()