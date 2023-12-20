import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


LENGTH = 20
BREADTH = 5
EPOCHS = 50
SIZE = LENGTH*BREADTH
BATCH_SIZE = 9
DATA_SIZE = 99
PATH = './stored_models/'
train_data = './data/img/'

torch.set_printoptions(profile="full")


class Wide_FFNN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.hidden = nn.Linear(size, 4*size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(4*size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
    
class Deep_FFNN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layer1 = nn.Linear(size, 2*size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(2*size, 2*size)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(2*size, size)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    

def trainer(num_epochs, model):
    start_time = time.time()

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 0.0001
    model.train()

    for epoch in range(num_epochs):
        step = 0
        tot = 0
        in_tensor = torch.zeros((BATCH_SIZE, SIZE))
        out_tensor = torch.zeros((BATCH_SIZE, 1))
        in_tot = torch.zeros((DATA_SIZE, SIZE))
        out_tot = torch.zeros((DATA_SIZE, 1))

        for stock in os.listdir(train_data):
            data_point = pd.read_csv(f'{train_data+stock}').tail(SIZE+1)['0'].values.tolist()
            in_vec = torch.FloatTensor(data_point[:-1])
            pos = data_point[-1]
            in_tensor[step] = in_vec[:]
            out_tensor[step] = pos
            in_tot[tot] = in_vec[:]
            out_tot[tot] = pos
            step += 1
            tot += 1

            if step == BATCH_SIZE:
                # forward pass
                out_pred = model(in_tensor)
                loss = loss_fn(out_pred, out_tensor)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

                step = 0
                in_tensor = torch.zeros((BATCH_SIZE, SIZE))
                out_tensor = torch.zeros((BATCH_SIZE, 1))


        # forward pass
        tot_pred = model(in_tot)
        loss = loss_fn(tot_pred, out_tot)
        # print progress
        acc = (tot_pred.round() == out_tot).float().mean()
        print(f'Total loss = {loss}, Total accuracy = {acc}, Epoch:{epoch+1} ended. Took {time.time()-start_time} seconds.')
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epochs': num_epochs
                }, os.path.join(PATH, 'deep.chk'))


# light_model = Wide_FFNN(SIZE)
# trainer(EPOCHS, light_model)
heavy_model = Deep_FFNN(SIZE)
trainer(EPOCHS, heavy_model)