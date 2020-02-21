def lstm_model(input_shape):
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(filters=256,kernel_size=15,strides=3)(X_input)                                 
    X = BatchNormalization()(X)                               
    X = Activation("relu")(X)                                 
    X = Dropout(0.2)(X)                                 
    
    X = GRU(32, return_sequences=True)(X)                                 
    X = Dropout(0.2)(X)                              
    X = BatchNormalization()(X)                                 
    
    X = GRU(32, return_sequences=True)(X)                                 
    X = BatchNormalization()(X)                                 
    X = Dropout(0.2)(X) 
    
    X = GRU(32, return_sequences=True)(X)                                 
    X = BatchNormalization()(X)                                 
    X = Dropout(0.2)(X) 
    
    X = GRU(32, return_sequences=True)(X)                                 
    X = BatchNormalization()(X)                                 
    X = Dropout(0.2)(X) 
    
    X = GRU(64, return_sequences=True)(X)                                 
    X = BatchNormalization()(X)                                 
    X = Dropout(0.2)(X)    
    
    X = GRU(64, return_sequences=True)(X)                                 
    X = BatchNormalization()(X)                                 
    X = Dropout(0.2)(X)                                 
    
    X = Dense(1, activation = "sigmoid")(X)
    X = Reshape((1595,))(X)
    
    model = Model(inputs = X_input, outputs = X)
    
    
    return model  
