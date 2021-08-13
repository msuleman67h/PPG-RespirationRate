import torch
import torch.nn as nn
import cnn
import ppg_dataset
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ppg, resp_rate, resp_points = ppg_dataset.training_data()

# Shaping the input
ppg = torch.from_numpy(ppg).float().to(device=device)
resp_rate = torch.from_numpy(resp_rate).float().to(device=device)

n_samples, n_features = ppg.t().shape
print(n_samples, n_features)

my_cnn = cnn.CNN()
my_cnn.to(device=device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(my_cnn.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    temp = torch.cat((ppg, resp_rate), dim=1)
    temp = temp[torch.randperm(temp.size()[0])]
    ppg = temp[:, 0:4000]
    resp_rate = temp[:, 4000:4001]
    for i, x in enumerate(ppg):
        if math.isnan(resp_rate[i].item()):
            continue

        # Forward pass
        outputs = my_cnn(x.unsqueeze(0).unsqueeze(0))
        loss = criterion(outputs, resp_rate[i].view(-1, 1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {num_epochs}, iteration: {i}, loss: {loss.item():.4f}")

indx = [i if math.isnan(x) else -1 for i, x in enumerate(resp_rate)]
for item in indx:
    if item != -1:
        print(f"Estimate: {my_cnn(ppg[item].unsqueeze(0).unsqueeze(0)).item()}, Actual: {resp_rate[item].item()}")