import time

import torch
import torch.nn as nn
import ppg_dataset
import math
import my_neural_net
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ppg, resp_rate, resp_points, resp_sig = ppg_dataset.training_data()

# Shaping the input
ppg = torch.from_numpy(ppg).float().to(device=device)
resp_rate = torch.from_numpy(resp_rate).float().to(device=device)
resp_points = torch.from_numpy(resp_points).float().to(device=device)
resp_sig = torch.from_numpy(resp_sig).float().to(device=device)

n_samples, n_features = ppg.t().shape

my_nn = my_neural_net.MyNeuralNet(sequence_length=125, output_size=1, input_size=32)
my_nn.to(device=device)

criterion = nn.L1Loss()
optimizer = torch.optim.SGD(my_nn.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    temp = torch.cat((ppg, resp_sig, resp_rate), dim=1)
    temp = temp[torch.randperm(temp.size()[0])]
    ppg = temp[:, 0:4000]
    resp_sig = temp[:, 4000:8000]
    resp_rate = temp[:, 8000:8001]
    for i, x in enumerate(ppg):
        if math.isnan(resp_rate[i].item()):
            continue

        x = x.view(-1, 125, 32)
        # Forward pass
        outputs = my_nn(x)
        target = resp_rate[i]
        loss = criterion(outputs, target.unsqueeze(0))

        # Backward and optimize
        print(f"Epoch: {epoch}, iteration: {i}, loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("ok")
indx = [i if math.isnan(x) else -1 for i, x in enumerate(resp_rate)]
for item in indx:
    if item != -1:
        result = my_nn(ppg[item].view(-1, 125, 32))
        print(result.item())
        # plt.plot(result[0].cpu().detach().numpy())
        # plt.plot(resp_sig[item].cpu().detach().numpy())
        # plt.clf()

