import numpy
import torch
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import medfilt, butter, cheby2, filtfilt


class BiomedicalSignals:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def butter_lowpass_filter(data: numpy.ndarray, cutoff: int, fs: int = 125, order: int = 5):
        b, a = butter(order, cutoff, fs=fs, btype='low')
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def cheby2_bandpass_filter(signal: numpy.ndarray, lowcut: float, highcut: float, fs: int = 125, order: int = 4):
        b, a = cheby2(order, 20, [lowcut, highcut], 'bandpass', fs=fs)
        y = filtfilt(b, a, signal)
        return y

    def __init__(self):
        self.raw_ppg = torch.empty(0)
        self.filtered_ppg = torch.empty(0)
        self.ppg_low_f = torch.empty(0)
        self.resp_sig = torch.empty(0)
        self.respiration_rate = torch.empty(0)
        self.breathing_annotation = torch.empty(0)
        self.data_len = 60001  # Roughly 8 mins of data

    def set_ppg(self, ppg: numpy.ndarray):
        """
        Saves the ppg in raw, low freq (0 - 1Hz) and filtered (1 - 15Hz) form
        :param ppg: Photoplethysmogram
        """
        self.data_len = ppg.size
        self.raw_ppg = torch.from_numpy(
            numpy.interp(ppg, (ppg.min(), ppg.max()), (-10, 10))).float().to(BiomedicalSignals.device)

        ppg_low = BiomedicalSignals.butter_lowpass_filter(ppg, cutoff=1)
        ppg_low = medfilt(ppg_low, kernel_size=145)
        ppg_low = uniform_filter1d(ppg_low, size=50)
        ppg_low = numpy.interp(ppg_low, (ppg_low.min(), ppg_low.max()), (-10, 10)) * -1
        self.ppg_low_f = torch.from_numpy(ppg_low).float().to(BiomedicalSignals.device)

        self.filtered_ppg = torch.from_numpy(numpy.ascontiguousarray(
            BiomedicalSignals.cheby2_bandpass_filter(ppg, 1, 15))).float().to(BiomedicalSignals.device)

    def set_resp_signal(self, resp_signal: numpy.ndarray):
        """
        Saves the respiration signal from device like Impedance Pneumography
        :param resp_signal: Respiration Signal
        """
        resp_signal = resp_signal
        resp_signal = numpy.interp(resp_signal, (resp_signal.min(), resp_signal.max()), (-10, 10))
        self.resp_sig = torch.from_numpy(resp_signal).float().to(BiomedicalSignals.device)

    def set_breathing_annotation(self, annotations: numpy.ndarray):
        """
        Saves the breathing annotation in the form of probability distribution
        :param annotations: Breathing Annotations
        """
        self.breathing_annotation = torch.zeros(self.data_len).to(BiomedicalSignals.device)

        # collecting the breathing points between 0 and data_len
        for col in annotations:
            if col > self.data_len:
                break
            self.breathing_annotation[numpy.int(col)] = 1

    def set_resp_rate(self, resp_rate: numpy.ndarray):
        """
        Sets the mean respiration rate for each minutes
        :param resp_rate: respiration rate
        """
        self.respiration_rate = torch.from_numpy(resp_rate).float().to(BiomedicalSignals.device)
