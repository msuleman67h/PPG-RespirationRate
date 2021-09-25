import numpy
import torch
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import medfilt, butter, cheby2, filtfilt


class BiomedicalSignals:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def butter_lowpass_filter(data: numpy.ndarray, cutoff: float, fs: int, order: int = 4):
        b, a = butter(order, cutoff, fs=fs, btype='low')
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def butter_highpass_filter(data: numpy.ndarray, cutoff, fs: int, order: int = 4):
        b, a = butter(order, cutoff, fs=fs, btype='high')
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def cheby2_lowpass_filter(signal: numpy.ndarray, cutoff: float, fs: int, order: int = 4):
        b, a = cheby2(order, 20, cutoff, 'low', fs=fs)
        y = filtfilt(b, a, signal)
        return y

    def __init__(self, sampling_rate, kernel_size):
        self.raw_ppg = torch.empty(0)
        self.filtered_ppg = torch.empty(0)
        self.extracted_resp_sig = torch.empty(0)
        self.resp_sig = torch.empty(0)
        self.respiration_rate = torch.empty(0)
        self.breathing_annotation = torch.empty(0)
        self.sampling_rate = sampling_rate
        self.kernel_size = kernel_size
        self.data_len = sampling_rate * 8 * 60 + 1

    def set_ppg(self, ppg: numpy.ndarray):
        """
        Saves the ppg in raw, low freq (0 - 1Hz) and filtered (1 - 15Hz) form
        :param ppg: Photoplethysmogram
        """
        self.raw_ppg = torch.from_numpy(
            numpy.interp(ppg, (numpy.min(ppg), numpy.max(ppg)), (-10, 10))).float().to(BiomedicalSignals.device)

        ext_resp_sig = BiomedicalSignals.cheby2_lowpass_filter(ppg, cutoff=1, fs=self.sampling_rate)
        ext_resp_sig = medfilt(ext_resp_sig, kernel_size=self.kernel_size)
        ext_resp_sig = uniform_filter1d(ext_resp_sig, size=50)
        ext_resp_sig = numpy.interp(ext_resp_sig, (numpy.min(ext_resp_sig), numpy.max(ext_resp_sig)), (-10, 10)) * -1
        self.extracted_resp_sig = torch.from_numpy(ext_resp_sig).float().to(BiomedicalSignals.device)

        ppg_filt = numpy.ascontiguousarray(BiomedicalSignals.butter_highpass_filter(ppg, 1, fs=self.sampling_rate))
        ppg_filt = numpy.interp(ppg_filt, (numpy.min(ppg_filt), numpy.max(ppg_filt)), (-10, 10))
        self.filtered_ppg = torch.from_numpy(ppg_filt).float().to(BiomedicalSignals.device)

    def set_resp_signal(self, resp_signal: numpy.ndarray):
        """
        Saves the respiration signal from device like Impedance Pneumography
        :param resp_signal: Respiration Signal
        """
        resp_signal = resp_signal
        resp_signal = numpy.interp(resp_signal, (numpy.min(resp_signal), numpy.max(resp_signal)), (-10, 10))
        self.resp_sig = torch.from_numpy(resp_signal).float().to(BiomedicalSignals.device)

    def set_breathing_annotation(self, annotations: numpy.ndarray):
        """
        Saves the breathing annotation in the form of probability distribution
        :param annotations: Breathing Annotations
        """

        self.breathing_annotation = torch.from_numpy(annotations).to(self.device)
        # self.breathing_annotation = torch.zeros(self.data_len).to(BiomedicalSignals.device)
        #
        # # collecting the breathing points between 0 and data_len
        # for col in annotations:
        #     if col > self.data_len:
        #         break
        #     self.breathing_annotation[numpy.int(col)] = 1

    def set_resp_rate(self, resp_rate: numpy.ndarray):
        """
        Sets the mean respiration rate for each minutes
        :param resp_rate: respiration rate
        """
        self.respiration_rate = torch.from_numpy(resp_rate).float().to(BiomedicalSignals.device)
