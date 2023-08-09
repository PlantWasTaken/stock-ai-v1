def seq():
    model = keras.Sequential([
            layers.LSTM(64,return_sequences=True),
            layers.LSTM(64,return_sequences=False),
            layers.Dense(1, name="layer3")
    ])
    
    model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
        )
    
    x = tf.ones((BATCH_SIZE, 15, 12))
    y = model(x)
    
    model.summary()
    model.fit(
        x=feat,
        y=lab,
        batch_size=None,
        epochs=10
    )