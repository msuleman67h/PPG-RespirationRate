import torch
import torch.nn as nn
import ppg_dataset
import math
import cnn_2d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppg, resp_rate, resp_points, resp_sig = ppg_dataset.training_data()

    # Shaping the input
    ppg = torch.from_numpy(ppg).float().to(device=device)
    resp_rate = torch.from_numpy(resp_rate).float().to(device=device)
    resp_points = torch.from_numpy(resp_points).float().to(device=device)

    n_samples, n_features = ppg.t().shape

    cnn = cnn_2d.CNN2D()
    cnn.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        temp = torch.cat((ppg, resp_points, resp_rate), dim=1)
        temp = temp[torch.randperm(temp.size()[0])]
        ppg = temp[:, 0:4096]
        resp_points = temp[:, 4096:8192]
        resp_rate = temp[:, 8192:8193]

        for i, x in enumerate(ppg):
            if math.isnan(resp_rate[i].item()):
                continue

            x = x.view(-1, 1, 64, 64)
            # Forward pass
            outputs = cnn(x)
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
            result = cnn(ppg[item].view(-1, 1, 64, 64))
            print(result.item())
            # plt.plot(result[0].cpu().detach().numpy())
            # plt.plot(resp_sig[item].cpu().detach().numpy())
            # plt.clf()
