import pandas as pd
from tqdm import tqdm
import plotly.express as px
import yfinance as yf
from datetime import date, timedelta

ticker = '5253.T'
interval = "2m"
start = date.today() - timedelta(days=7)
end = date.today()

data = yf.download(ticker, start=start, end=end,interval=interval)
df = pd.DataFrame(data)

print(df)

#  Date       | Open   | High   | Low    | Close  | #Adj Close | Volume #=del
#df = df.drop(columns=['Date'])
del df['Adj Close'] #adj close
del df['Open']
del df['High']
del df['Low']
del df['Volume']

#0 indicated neutral -- all values relate to closing
#df['num'] = [i for i in range(len(df))]
df['mov'] = '' #1 = rising, -1 falling -- closing price
df['5sMvA'] = ''
df['8sMvA'] = ''
df['13sMvA'] = ''
df['5wMvA'] = ''
df['8wMvA'] = ''
df['13wMvA'] = ''

mov = []
for i in tqdm(range(len(df['Close']))):
    try:
        val = df['Close'][i] - df['Close'][i-1]
        if(val < 0):
            mov.append(-1)
        else:
            mov.append(1)
    except:
        mov.append(0)

def sma(v):
    mva = [(sum(list(df['Close'][i:i+v]))/v) for i in tqdm(range(len(df['Close'])-v))]
    mva[0:0] = [0 for _ in range(v)]
    return mva

def wma(v):
    wma_l = []
    for i in tqdm(range(len(df['Close'])-v)):
        val = list(df['Close'][i:i+v])
        val = sum([((val[j]*(v-j))) for j in range(len(val))])/sum(h for h in range(v+1))
        wma_l.append(val)
    wma_l[0:0] = [0 for _ in range(v)]
    return wma_l

df['mov'] = mov

df['5sMvA'] = sma(5)
df['8sMvA'] = sma(8)
df['13sMvA'] = sma(13)

df['5wMvA'] = wma(5)
df['8wMvA'] = wma(8)
df['13wMvA'] = wma(13)

df = df.iloc[13:] #dropping bad data

def plot():
    #removing "break time"
    #for i in df

    #exit()
    lst = [df['5sMvA'],df['8sMvA'],df['13sMvA'], df['5wMvA'], df['8wMvA'], df['13wMvA'], df['Close']]
    #lst = [df['5wMvA'], df['8wMvA'], df['13wMvA'], df['close'] ]
    #fig = px.line(df, x=df['num'],y=lst, title='Line Graph for Two Lines')

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=2, cols=1)

    fig.append_trace(go.Line(
        x=df['num'],
        y=df['Close'],
    ), row=1, col=1)

    fig.append_trace(go.Line(
        x=df['num'],
        y=df['Volume'],
    ), row=2, col=1)





df.to_csv('test.csv', encoding='utf-8', index=False)
print(df)