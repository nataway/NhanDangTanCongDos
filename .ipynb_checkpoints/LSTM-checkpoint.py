# %%
import pandas as pd
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# %%
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %%
path = r"/Users/taanhtuan/Desktop/workproject/ids-2018-detection/02-15-2018.csv"

# %%
df = pd.read_csv(path)

# %%
df.head()

# %%
for col in df.columns:
    print(col, ": ", df.iloc[0][col])

# %%
num_of_samples = len(df)
unique_values = df["Dst Port"].value_counts()
unique_values = unique_values.reset_index()
for value in unique_values.iterrows():
#     percentage = float((value[1]["Dst Port"]*100/num_of_samples))
#     if percentage > 0.01:
#         print("the percentage of ", value[1]['index'], " occupied", percentage)
# print("and some values with trivial percentage")
    print("the number of ", value[1]['index'], " is", value[1]["Dst Port"])

# %%
df.sort_values(by=["Dst Port"])

# %%
dropped_columns = ["Timestamp"]
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
heig = len(X.columns)
Dst_list = []
X_uniq_vals = X["Dst Port"].value_counts()
X_uniq_vals = X_uniq_vals.reset_index()

leng = X_uniq_vals.iloc[0]['Dst Port']

print("leng: ", leng)
print("heig: ", heig)

# %%
inputs = np.zeros((1500000,100,78))
num_of_inputs = 0

his_record = 100
num_of_feat = 78
    
for value in X_uniq_vals.iterrows():
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
        if(num_of_inputs == 10000):
            break;

    if value[1]['index'] == 53:
        break;
#         arr = torch.from_numpy(arr).float()

# %%
inputs = torch.from_numpy(inputs[:num_of_inputs])

# %%
labels = torch.from_numpy(Y[:num_of_inputs])

# %%
x_train, x_test, y_train, y_test = train_test_split(inputs, labels,test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.2, random_state=42)

# %%
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
test_data = TensorDataset(x_test, y_test)

batch_size = 100

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# %%
class DosDetection(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.5):
        super(DosDetection, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.float()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
#         return 1, 1
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hid1 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        hid2 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        hidden = (hid1.float(), hid2.float())
        return hidden

# %%
input_size = 78
hidden_dim = 32
n_layers = 100
output_size = 3

model = DosDetection(input_size, output_size, hidden_dim, n_layers)
model.to(device)
print(model)

# %%
lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %%
epochs = 2
counter = 0
print_every = 1
clip = 5
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    counter = 0
    h = model.init_hidden(batch_size)
    
    for idx, (inputs, labels) in enumerate(train_loader):
        counter += 1
        h = tuple([e.data.float() for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if counter%print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
                
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

# %%
# Loading the best model
model.load_state_dict(torch.load('./state_dict.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))

# %%
