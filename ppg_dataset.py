import os

import mat73
import numpy
import scipy.io as spio
import scipy.signal
import matplotlib.pyplot as plt

import bio_signals


def training_data():
    # loading bidmc dataset
    data = load_mat('data/bidmc/bidmc_data.mat')['data']

    training_dataset = []

    for row in data:
        bio = bio_signals.BiomedicalSignals(125, 195)

        bio.set_resp_rate(row.ref.params.rr.v[:480].reshape(60, -1).mean(axis=0))
        bio.set_ppg(row.ppg.v)
        bio.set_resp_signal(row.ref.resp_sig.imp.v)

        # Getting Resp annotated points
        # Making sure both the breathing annotations are of same size
        ann1 = row.ref.breaths.ann1
        ann2 = row.ref.breaths.ann2
        min_size = min(numpy.size(ann1), numpy.size(ann2))
        ann1 = ann1[:min_size]
        ann2 = ann2[:min_size]
        bio.set_breathing_annotation(numpy.stack((ann1, ann2)).mean(axis=0).astype(numpy.int))
        training_dataset.append(bio)

    # loading capnobase dataset
    # capno_base_data = []
    # for file in os.listdir("data/capnobase"):
    #     if file.endswith(".mat"):
    #         data_dict = mat73.loadmat(f"data/capnobase/{file}")
    #         capno_base_data.append(data_dict)
    #
    #         bio = bio_signals.BiomedicalSignals(300, 499)
    #         time_axis = data_dict['reference']['rr']['co2']['x']
    #         breathing_rate = data_dict['reference']['rr']['co2']['y']
    #         minute = 1
    #         instant_resp_rate = 0
    #         resp_rates = []
    #         prev_index = 0
    #         for index, value in numpy.ndenumerate(time_axis):
    #             if value >= minute * 60:
    #                 instant_resp_rate = instant_resp_rate / (index[0] + 1 - prev_index)
    #                 prev_index = index[0]
    #                 minute += 1
    #                 resp_rates.append(instant_resp_rate)
    #                 instant_resp_rate = breathing_rate[index]
    #             else:
    #                 instant_resp_rate += breathing_rate[index]
    #
    #         bio.set_resp_rate(numpy.asarray(resp_rates))
    #         bio.set_ppg(data_dict['signal']['pleth']['y'])
    #         bio.set_resp_signal(data_dict['signal']['co2']['y'])
    #
    #         ann1 = data_dict['labels']['co2']['startexp']['x']
    #         ann2 = data_dict['labels']['co2']['startinsp']['x']
    #         min_size = min(numpy.size(ann1), numpy.size(ann2))
    #         ann1 = ann1[:min_size]
    #         ann2 = ann2[:min_size]
    #
    #         z = numpy.stack((ann1, ann2)).mean(axis=0)
    #         bio.set_breathing_annotation(z.astype(numpy.int))
    #         training_dataset.append(bio)

    return training_dataset


def load_mat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
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
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
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