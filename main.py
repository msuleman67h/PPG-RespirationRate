import csv
import torch
import cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("ppg.csv") as fp1, open('breaths.csv') as fp2:
    reader = csv.reader(fp1)
    ppg = [[float(item) for item in row[:1250]] for row in reader]
    reader = csv.reader(fp2)
    breaths = [[int(item) for item in row if item != ''] for row in reader]

# Shaping the input
ppg = torch.tensor(ppg, device=device)
ppg = torch.reshape(ppg, [53, 1250, 1])
print(ppg.size())

my_cnn = cnn.CNN()
my_cnn.to(device=device)
my_cnn.forward(ppg)
