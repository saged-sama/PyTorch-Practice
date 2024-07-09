import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
# %matplotlib inline

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(0)
model = Model()

url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_df = pd.read_csv(url)

my_df['species'] = my_df['species'].replace({'setosa': 0.0, 'versicolor': 1.0, 'virginica': 2.0})


x_axis = my_df.drop('species', axis=1)
y_axis = my_df['species']

x_axis = x_axis.values
y_axis = y_axis.values

x_train, x_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.2, random_state=0)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)   

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
losses = []

for i in range(epochs):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    # if i % 10 == 0:
    #     print(f"Epoch: {i} and loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(losses[-1])

with torch.no_grad():
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)

correct = 0
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model.forward(data)

        print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(correct)

plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("epochs")
plt.show()

torch.save(model.state_dict(), "flower_model.pt")