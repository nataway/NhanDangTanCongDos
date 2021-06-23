# %%
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model, Model
from attention import Attention

# %%
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code. "gpu"/"tpu"
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %%
path = r"/Users/taanhtuan/Desktop/ids-2018-demo/combined_data.csv"

# %%
df = pd.read_csv(path)

# %%
df

# %%
for col in df.columns:
    print(col, ": ", df.iloc[0][col])

# %%
for col in df.columns:
    num_of_samples = len(df)
    unique_values = df[col].value_counts()
    unique_values = unique_values.reset_index()
    for value in unique_values.iterrows():
        percentage = float((value[1][col]*100/num_of_samples))
        if percentage > 99:
            print("Col: ", col)
            print("the percentage of ", value[1]['index'], " occupied", percentage)
        else:
            break;
#     print("and some values with trivial percentage")

# %%
num_of_samples = len(df)
unique_values = df["Dst Port"].value_counts()
unique_values = unique_values.reset_index()
for value in unique_values.iterrows():
    percentage = float((value[1]["Dst Port"]*100/num_of_samples))
    if percentage > 0.01:
        print("the percentage of ", value[1]['index'], " occupied", percentage)
print("and some values with trivial percentage")

# %%
df.sort_values(by=["Dst Port", "Timestamp"])

# %%
dropped_columns = ["Timestamp", "Bwd PSH Flags", "Bwd URG Flags", "FIN Flag Cnt", "CWE Flag Count", "Fwd Byts/b Avg",
                  "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"]
df = df.drop(dropped_columns, axis=1)

# %%
X = df.iloc[:, df.columns != 'Label']
Y = df.iloc[:, df.columns == 'Label']

# %%
Y = df.iloc[:, df.columns == 'Label']
unique_values = Y["Label"].value_counts()
print(unique_values)

# %%
ord_enc = OrdinalEncoder()
Y = ord_enc.fit_transform(Y)

# %%
Y = Y.ravel()

# %%
Y.shape

# %%
X

# %%
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()

# %%
heig = len(X.columns)
Dst_list = []
X_uniq_vals = X["Dst Port"].value_counts()
X_uniq_vals = X_uniq_vals.reset_index()

leng = X_uniq_vals.iloc[0]['Dst Port']

print("leng: ", leng)
print("heig: ", heig)

# %%
inputs = np.zeros((1050000 ,20,68))
num_of_inputs = 0
num_of_inputsVal = 0

his_record = 20
num_of_feat = 68
    
for value in X_uniq_vals.iterrows():
    num_of_inputsVal = 0
    print("value index: ", value[1]['index'])
    
    mask = X['Dst Port'] == value[1]['index']
    dfX = X[mask].to_numpy()
    print(type(dfX))
    dfX = dfX.transpose()
    
    extra_df = np.zeros((num_of_feat, his_record))
    count = 0;
    
    new_df = np.concatenate((extra_df, dfX), axis=1)
    new_df = new_df.transpose()
    
    df_leng = new_df.shape[0]
    
    print(extra_df.shape)
    print(dfX.shape)
    print(new_df.shape)
    print(new_df[0].shape)
    print(df_leng)
    
    for i in range(his_record,df_leng):
        print("num_of_inputs: ", num_of_inputs)
        inputs[num_of_inputs] = new_df[i-his_record:i][:]
        num_of_inputs += 1
        num_of_inputsVal += 1
        if(num_of_inputsVal == 10000):
            break;

#     if value[1]['index'] == 53:
#         break;
#         arr = torch.from_numpy(arr).float(

# %%
inputs = inputs[:num_of_inputs]

# %%
labels = Y[:num_of_inputs]

# %%
print(type(labels))
print(type(inputs))
print()
print(labels.shape)
print(inputs.shape)

# %%
x_train, x_test, y_train, y_test = train_test_split(inputs, labels,test_size=0.15, random_state=42)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.2, random_state=42)

# %%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# %%
numofclass = np.unique(labels, return_inverse=False)
print(numofclass)

# %%
# Dummy data. There is nothing to learn in this example.
num_samples = 193721

time_steps = 20
input_dim = 68

output_dim = 1

print("size: ", num_samples, " ", time_steps, " ", input_dim)

data_x = x_train
data_y = y_train

# %%
print("ABC")

# Define/compile the model.
model_input = Input(shape=(time_steps, input_dim))
print("model_input: ", model_input.shape) # input(None, 20, 78)
x = LSTM(64, return_sequences=True)(model_input)
print(x.shape) # outputLSTM(None, 20, 64)
x = Dropout(0.5)(x)
print(x.shape) # outputDropout(None, 20, 64)
x = Attention(32)(x)
print(x.shape) # outputAttention(None, 32)
initializer = tf.keras.initializers.GlorotUniform(seed=42)
activation =  None
x = Dense(3, kernel_initializer=initializer, activation=activation)(x)
print(x.shape) # output(None, 1)

model = Model(model_input, x)
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # default from_logits=False
              metrics=[keras.metrics.SparseCategoricalAccuracy()])



# %%
print(model.summary())

# train.
model.fit(data_x, data_y, epochs=10)

# test save/reload model.
pred1 = model.predict(data_x)
model.save('test_model.h5')


# %%
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model, Model

from attention import Attention

time_steps = 20
input_dim = 68

# Define/compile the model.
model_input = Input(shape=(time_steps, input_dim))
x = LSTM(64, return_sequences=True)(model_input)
x = Dropout(0.5)(x)
x = Attention(32)(x)
x = Dense(3)(x)


model = Model(model_input, x)
model.compile(loss='bce', optimizer='adam')
print(model.summary())

model_h5 = load_model('test_model.h5')
pred_muilt = model_h5.predict(x_test)
    

# %%
print(pred_muilt.shape)
preTest = np.zeros(32999)
for i in range(pred_muilt.shape[0]):
    preTest[i] = np.argmax(pred_muilt[i], axis=0)

numCorrect = 0
print(preTest.shape)
print(y_test.shape)
for i in range(preTest.shape[0]):
    if y_test[i] == preTest[i]:
        numCorrect += 1
print("Accuracy: ", numCorrect/preTest.shape[0])

# %%
