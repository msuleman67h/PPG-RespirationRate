import numpy
import scipy.io as spio
from scipy.signal import medfilt, cheby2, filtfilt
import matplotlib.pyplot as plt


def bandpass_filter(signal, lowcut=0.1, highcut=10.0):
    fs = 125
    fn = fs * 0.5
    b, a = cheby2(4, 20, [lowcut / fn, highcut / fn], 'bandpass')
    y = filtfilt(b, a, signal)
    return y


def training_data():
    # loading mat file
    data = load_mat('data/bidmc_data.mat')['data']

    data_len = 4096
    # Reading ppg data from mat file
    ppg = numpy.zeros(shape=(53, data_len))
    resp_sig = numpy.zeros(shape=(53, data_len))
    respiration_rate = numpy.zeros(shape=(53, 1))
    ppg_breath_points = numpy.zeros(shape=(53, data_len))

    for index, row in numpy.ndenumerate(data):
        respiration_rate[index] = row.ref.params.rr.v.mean()

    for index, row in numpy.ndenumerate(data):
        ppg_row = medfilt(row.ppg.v, kernel_size=5)
        # Discarding the first 500 rows to omit the spike due to filter
        ppg[index] = bandpass_filter(ppg_row)[0:data_len]
        ppg[index] = numpy.power(ppg[index], 2)
        ppg[index] = numpy.interp(ppg[index], (ppg[index].min(), ppg[index].max()), (0, 1))

        resp_sig[index] = row.ref.resp_sig.imp.v[0:data_len]
        resp_sig[index] = numpy.interp(resp_sig[index], (resp_sig[index].min(), resp_sig[index].max()), (0, 1))

    # Reading breathing data from mat file
    i = 0
    for row in data:
        # Making sure both the breathing annotations are of same size
        ann1 = row.ref.breaths.ann1
        ann2 = row.ref.breaths.ann2
        min_size = min(numpy.size(ann1), numpy.size(ann2))
        ann1 = ann1[:min_size]
        ann2 = ann2[:min_size]

        # Marking the points in PPG at which the person inhaled
        breaths = numpy.mean([ann1, ann2], axis=0)[:40]
        # collecting the breathing points between 0 and data_len
        for col in breaths:
            if col > data_len:
                break
            ppg_breath_points[i, numpy.int(col)] = 1

        i += 1
    return ppg, respiration_rate, ppg_breath_points, resp_sig


def load_mat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, numpy.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, numpy.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

# n = len(ppg[i])  # length of the signal
# k = numpy.arange(n)
# T = n / 125
# frq = k / T  # two sides frequency range
# frq = frq[:len(frq) // 2]  # one side frequency range
#
# Y = numpy.fft.fft(ppg[i]) / n  # dft and normalization
# Y = Y[:n // 2]
# plt.plot(frq, abs(Y))
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|Y(freq)|')
