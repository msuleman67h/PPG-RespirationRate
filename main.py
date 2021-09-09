import math
import random

import scipy.signal
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy

import my_neural_net
import ppg_dataset


def train_nn():
    """
    Trains the neural network on the first minute data from ppg
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.05)

    num_epochs = 2
    for epoch in range(num_epochs):
        random.shuffle(training_dataset)
        for i, bio_signal in enumerate(training_dataset):
            x = bio_signal.ppg_low_f[0:7500]
            my_nn_out = my_nn(x)

            target = bio_signal.breathing_annotation[0:7500]
            # The commented lines below are needed if we want to use CrossEntropyLoss
            # target = target.nonzero()
            # my_nn_out = my_nn_out.repeat(target.size()[0], 1)
            loss = criterion(my_nn_out, target)

            # Backward and optimize
            print(f"Epoch: {epoch}, iteration: {i}, loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test_nn():
    """
    Tests the neural network on data from second minute
    """
    for i, bio_signal in enumerate(training_dataset):
        x = bio_signal.ppg_low_f[7500:15000].view(1, 250, -1)
        result = my_nn(x)
        plt.plot(result.cpu().detach().numpy())
        plt.plot(bio_signal.breathing_annotation[7500:15000].cpu().detach().numpy())
        plt.plot(bio_signal.ppg_low_f[7500:15000].cpu().detach().numpy())
        plt.show()
        plt.clf()


def identify_peaks():
    """
    Identifies the respiration points with conventional methods
    """
    for i, bio_signal in enumerate(training_dataset):
        z = bio_signal.ppg_low_f[:7500].cpu().detach().numpy()
        peaks, p_height = scipy.signal.find_peaks(z, height=numpy.median(z))
        plt.plot(z)
        plt.plot(bio_signal.resp_sig[:7500].cpu().detach().numpy())
        plt.plot(peaks, z[peaks], "or")
        # y = bio_signal.raw_ppg[:3000].cpu().detach().numpy()
        # plt.plot(y)
        plt.plot(bio_signal.breathing_annotation[:7500].cpu().detach().numpy())
        plt.show()
        plt.clf()

        # x = bio_signal.ppg_low_f[0:5000].cpu().detach().numpy()
        # plt.plot(x)
        # plt.plot(bio_signal.resp_sig[0:5000].cpu().detach().numpy())
        # peaks, _ = find_peaks(x, height=numpy.median(x))
        # plt.plot(peaks, x[peaks], "or")
        # plt.plot(bio_signal.breathing_annotation[0:5000].cpu().detach().numpy())
        # plt.axhline(y=numpy.median(x), color='grey', linestyle='--')
        # plt.plot(bio_signal.ppg[0:5000].cpu().detach().numpy())
        # plt.show()
        # plt.clf()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_dataset = ppg_dataset.training_data()

    my_nn = my_neural_net.MyNeuralNet(125, 7500, 60, 40, 2).to(device)
    train_nn()
    test_nn()
