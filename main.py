import math
import random
import statistics

import numpy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

import my_neural_net
import ppg_dataset


def train_nn(my_nn, training_dataset):
    """
    Trains the neural network on the first minute data from ppg
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.1)

    num_epochs = 2
    for epoch in range(num_epochs):
        random.shuffle(training_dataset)
        for i, bio_signal in enumerate(training_dataset):
            for j in range(6):
                x = bio_signal.raw_ppg[j * 7500: (j + 1) * 7500]
                my_nn_out = my_nn(x)

                target = bio_signal.breathing_annotation[j * 7500: (j + 1) * 7500]
                # The commented lines below are needed if we want to use CrossEntropyLoss
                # target = target.nonzero().flatten()
                # my_nn_out = my_nn_out.repeat(target.size()[0], 1)
                # loss = criterion(my_nn_out, target)
                loss = criterion(my_nn_out, target.unsqueeze(0))

                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"Epoch: {epoch}, iteration: {i}, loss: {loss.item():.4f}")


def test_nn(my_nn, training_dataset):
    """
    Tests the neural network on data from second minute
    """
    for i, bio_signal in enumerate(training_dataset):
        x = bio_signal.raw_ppg[7500 * 7:7500 * 8]
        result = my_nn(x)
        plt.plot(result[0].cpu().detach().numpy())
        plt.plot(bio_signal.breathing_annotation[7500:15000].cpu().detach().numpy())
        plt.plot(bio_signal.resp_sig[7500 * 7:7500 * 8].cpu().detach().numpy())
        plt.plot(bio_signal.extracted_resp_sig[7500 * 7:7500 * 8].cpu().detach().numpy())
        plt.show()
        plt.clf()


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (numpy.diff(numpy.sign(numpy.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (numpy.diff(numpy.sign(numpy.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = numpy.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[[i + numpy.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + numpy.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


def bland_altman_plot(data1, data2, *args, **kwargs):
    print(f"Pearson Correlation Coefficient: {numpy.corrcoef(data1, data2)[1, 0].item()}")
    data1 = numpy.asarray(data1)
    data2 = numpy.asarray(data2)
    mean = numpy.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = numpy.mean(diff)  # Mean of the difference
    sd = numpy.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs, s=128)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.xlabel("Mean of Estimate and Ground Truth Breathing Rate", fontsize=38)
    plt.ylabel("Difference between Estimate\nand Ground Truth Breathing rate", fontsize=38)
    plt.tick_params(labelsize=32)
    plt.subplots_adjust(left=0.13, bottom=0.12, right=0.97, top=0.97)
    plt.show()
    plt.clf()


def algorithim_evaluation(training_dataset):
    """
    Calculates the Mean Absolute Error, Standard Deviation as well as plots the Bland Altman for the dataset

    :param training_dataset: PPG dataset
    """
    diff_est_actual_resp_rate = []
    estimate_respiration_rate = []
    actual_respiration_rate = []
    for i, bio_signal in enumerate(training_dataset):
        multiple_of_minute = bio_signal.sampling_rate * 60
        for j in range(0, 7):
            # Get the filtered ppg signal and estimate the breathing annotation running find peak algorithm
            extracted_resp_sig = bio_signal.extracted_resp_sig[j * multiple_of_minute: (j + 1) * multiple_of_minute].cpu().detach().numpy()
            peaks, _ = find_peaks(extracted_resp_sig)
            estimate_respiration_rate.append(len(peaks))

            # Get the manual annotation done by humans
            breathing_ann = bio_signal.breathing_annotation.cpu().detach().numpy()
            manual_annotations = \
            numpy.where(numpy.logical_and(breathing_ann >= j * multiple_of_minute, breathing_ann <= (j + 1) * multiple_of_minute))[
                0]
            manual_annotations = bio_signal.breathing_annotation.cpu().detach().numpy()[manual_annotations]
            actual_respiration_rate.append(len(manual_annotations))

            diff_est_actual_resp_rate.append(abs(estimate_respiration_rate[i] - actual_respiration_rate[i]))

    mean_absolute_err = sum(diff_est_actual_resp_rate) / len(diff_est_actual_resp_rate)
    standard_dev = sum(
        [pow((x - statistics.mean(diff_est_actual_resp_rate)), 2) for x in diff_est_actual_resp_rate]) / (
                           len(diff_est_actual_resp_rate) - 1)
    print(f"Mean Absolute Error: {mean_absolute_err}, Standard Deviation: {standard_dev}")

    bland_altman_plot(estimate_respiration_rate, actual_respiration_rate)


def compare_raw_ppg_and_filt_ppg(training_dataset):
    for i, bio_signal in enumerate(training_dataset):
        multiple_of_minute = bio_signal.sampling_rate * 60
        j = 0
        ext_resp = bio_signal.extracted_resp_sig[j * multiple_of_minute: (j + 1) * multiple_of_minute].cpu().detach().numpy()
        peaks, _ = find_peaks(ext_resp)

        raw_ppg = bio_signal.raw_ppg[j * multiple_of_minute: (j + 1) * multiple_of_minute].cpu().detach().numpy()

        fig, axs = plt.subplots(2, sharex=True)
        bg_ax = fig.add_subplot(111, frameon=False)
        bg_ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

        axs[0].plot(raw_ppg, 'tab:orange', label='PPG')
        axs[0].tick_params(labelsize=32)
        axs[0].locator_params(axis="y", nbins=6)
        ti = numpy.linspace(0, 60, 11)
        axs[0].set_xticks(ti * bio_signal.sampling_rate)
        axs[0].set_xticklabels(ti)

        axs[1].plot(ext_resp, color='tab:blue', label='Filtered PPG')
        axs[1].tick_params(labelsize=32)
        axs[1].locator_params(axis="y", nbins=6)
        axs[1].set_xticks(ti * bio_signal.sampling_rate)
        axs[1].set_xticklabels(ti)

        breathing_ann = bio_signal.breathing_annotation.cpu().detach().numpy()
        man_ann = numpy.where(numpy.logical_and(breathing_ann >= j * multiple_of_minute, breathing_ann <= (j + 1) * multiple_of_minute))[0]
        man_ann = bio_signal.breathing_annotation.cpu().detach().numpy()[man_ann]
        axs[1].plot(man_ann, ext_resp[man_ann], 'xg', markersize=10, label='Breathing Annotation')
        axs[1].plot(peaks, ext_resp[peaks], 'or', label='Peak Detection Output')

        bg_ax.set_xlabel("Time (s)", fontsize=38, labelpad=30)
        bg_ax.set_ylabel("Amplitude (mV)", fontsize=38, labelpad=60)
        plt.subplots_adjust(left=0.09, bottom=0.125, right=0.98, top=0.92, hspace=0.05)
        fig.legend(loc="upper center", mode="expand", ncol=4, fontsize=24)
        plt.show()


def random_sample(training_dataset, sample_size):
    """
    Takes random samples from the dataset and plots them in a single image

    :param training_dataset: PPG data
    :param sample_size: The number of unique sample to be included in plot
    """
    idxs = random.sample(range(0, len(training_dataset)), sample_size)

    fig, axs = plt.subplots(sample_size, sharex=True, sharey=True)
    bg_ax = fig.add_subplot(111, frameon=False)
    bg_ax_tx = bg_ax.twinx()
    axs2 = numpy.asarray([x.twinx() for x in axs])
    i = 0
    j = 0
    for index, row in enumerate(axs):
        bio_signal = training_dataset[idxs[i]]
        multiple_of_minute = bio_signal.sampling_rate * 60
        ext_resp = bio_signal.extracted_resp_sig[j * multiple_of_minute: (j + 1) * multiple_of_minute].cpu().detach().numpy()
        peaks, _ = find_peaks(ext_resp)
        row.plot(ext_resp, label='Filtered PPG Signal')
        row.tick_params(labelsize=24)
        row.locator_params(axis="y", nbins=6)

        axs2[index].plot(bio_signal.resp_sig[j * multiple_of_minute: (j + 1) * multiple_of_minute].cpu().detach().numpy(), '-.', label='Respiratory Signal', color='tab:orange')
        axs2[index].tick_params(labelsize=24)
        axs2[index].locator_params(axis="y", nbins=6)

        breathing_ann = bio_signal.breathing_annotation.cpu().detach().numpy()
        man_ann = numpy.where(numpy.logical_and(breathing_ann >= j * multiple_of_minute, breathing_ann <= (j + 1) * multiple_of_minute))[0]
        man_ann = bio_signal.breathing_annotation.cpu().detach().numpy()[man_ann]
        row.plot(man_ann, ext_resp[man_ann], 'xg', markersize=10, label='Breathing Annotation')
        row.plot(peaks, ext_resp[peaks], 'or', label='Peak Detection Output')

        i += 1
    bg_ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    bg_ax.set_xlabel("Time (s)", fontsize=38, labelpad=30)
    bg_ax.set_ylabel("Filtered PPG Signal (mV)", fontsize=38, labelpad=40, color='tab:blue')

    bg_ax_tx.set_ylabel(r'Respiratory Signal ($\Omega$)', color='orange', fontsize=38, labelpad=40)
    bg_ax_tx.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    lines, labels = axs[-1].get_legend_handles_labels()
    lines2, labels2 = axs2[-1].get_legend_handles_labels()
    lines = lines + lines2
    labels = labels + labels2
    plt.figlegend(lines, labels, loc="upper center", mode="expand", ncol=4, fontsize=24)
    plt.subplots_adjust(left=0.09, bottom=0.13, right=0.90, top=0.92, hspace=0.05)
    ti = numpy.linspace(0, 60, 11)
    plt.xticks(ti * bio_signal.sampling_rate, ti)
    plt.show()
    plt.clf()


def fourier_transform_ppg(training_dataset):
    """
    Performs Fourier Transform on pgg and plots them using matplotlib

    Works on BIDMC only for now
    
    :param training_dataset: PPG dataset loaded from files
    """
    for bio_signal in training_dataset:
        # For fourier analysis
        n = 60001  # length of the signal
        k = numpy.arange(n)
        T = n / 125
        frq = k / T  # two sides frequency range
        frq = frq[:len(frq) // 2]  # one side frequency range

        Y = numpy.fft.fft(bio_signal.raw_ppg.cpu().detach().numpy()) / n  # dft and normalization
        Y = Y[:n // 2]
        # We don't need frequencies beyond 2 Hz so it cuts them before plotting
        plt.plot(frq[:961], abs(Y[:961]))
        plt.axvline(1.0, color='grey', linestyle='--')
        plt.axvline(0.1, color='grey', linestyle='-.')
        plt.axvline(0.4, color='grey', linestyle='-.')
        plt.xticks(numpy.linspace(0, 2, 11))
        plt.xlabel('Frequency (Hz)', fontsize=42)
        plt.ylabel(r'Fast Fourier Transform ($V^2/Hz$)', fontsize=42)
        plt.tick_params(labelsize=36)
        plt.subplots_adjust(left=0.09, bottom=0.13, right=0.97, top=0.96)
        plt.show()
        plt.clf()


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # my_nn = my_neural_net.MyNeuralNet(250, 7500, 30, 40, 2).to(device)
    # train_nn(my_nn, training_dataset)
    # test_nn(my_nn, training_dataset)
    # identify_peaks(training_dataset)

    training_dataset = ppg_dataset.training_data()
    random_sample(training_dataset, 2)


if __name__ == '__main__':
    main()
