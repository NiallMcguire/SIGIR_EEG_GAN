import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)


def get_dataset():
    """
    :param:     - None
    :return:    - String of the link to the OpenMIIR-RawEEG Dataset
    """
    return "Data set can be found at >> http://www.ling.uni-potsdam.de/mlcog/OpenMIIR-RawEEG_v1"


def import_data(path):
    """
    :param:     - Path, this variable should be a string that contains the path to the fif file that the users wants to load, file must be .fif type
    :return:    - String of the link to the OpenMIIR-RawEEG Dataset
    """
    raw_file = mne.io.read_raw_fif(path)
    return raw_file


def to_pandas_dataframe(raw_file, sample_frequency, end):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function; sample_frequency, pass in the sample frequency of the data; end, pass in the end time of the data in seconds
    :return:    - A pandas Data frame containing the loaded in raw EEG file
    """

    start_end_secs = np.array([0, end])  # For this dataset data start at 0.000 and end at 4840.166 secs
    start_sample, stop_sample = (start_end_secs * sample_frequency).astype(int)
    data_frame = raw_file.to_data_frame(picks=['all'], start=start_sample, stop=stop_sample)
    return data_frame


def merge_trial_and_audio_onsets(raw_file, use_audio_onsets=True, inplace=True, stim_channel='STI 014', verbose=None):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - The Merged events list from the raw EEG file
    """
    events = mne.find_events(raw_file, stim_channel='STI 014', shortest_event=0)

    merged = list()
    last_trial_event = None
    for i, event in enumerate(events):
        etype = event[2]
        if etype < 1000 or etype == 1111:  # trial or noise onset
            if use_audio_onsets and events[i + 1][2] == 1000:  # followed by audio onset
                onset = events[i + 1][0]
                merged.append([onset, 0, etype])
                if verbose:
                    log.debug('merged {} + {} = {}'.format(event, events[i + 1], merged[-1]))
            else:
                # either we are not interested in audio onsets or there is none
                merged.append(event)
                if verbose:
                    log.debug('kept {}'.format(merged[-1]))
        # audio onsets (etype == 1000) are not copied
        if etype > 1111:  # other events (keystrokes)
            merged.append(event)
            if verbose:
                log.debug('kept other {}'.format(merged[-1]))

    merged = np.asarray(merged, dtype=int)

    if inplace:
        stim_id = raw_file.ch_names.index(stim_channel)
        raw_file._data[stim_id, :].fill(0)  # delete data in stim channel
        raw_file.add_events(merged)

    return merged


def get_events(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - The sequence events that occur within the raw EEG trial data, events structure (sample number, 0, event ID)
    """
    events = mne.find_events(raw_file, stim_channel='STI 014', shortest_event=0)
    return events


def print_info(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - Prints the info for the raw file
    """
    print("######################################")
    print(raw_file.info)
    print("######################################")


def print_bad_channels(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - Prints the band channels noted in the raw file
    """
    print("######################################")
    print("Bad Channels: ", raw_file.info['bads'])
    print("######################################")


def plot_easy_cap(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - Plots the easy cap diagrams
    """
    easycap_montage = mne.channels.make_standard_montage('biosemi64')
    print(easycap_montage)
    easycap_montage.plot()  # 2D
    fig = easycap_montage.plot(kind='3d', show=False)  # 3D
    fig = fig.gca().view_init(azim=70, elev=15)
    raw_file.set_montage('biosemi64')
    fig = raw_file.plot_sensors(show_names=True)


def plot_events(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - PLots the sequence of events across the raw data
    """
    events = get_events(raw_file)
    plt.figure(figsize=(17, 10))
    axes = plt.gca()
    mne.viz.plot_events(events, raw_file.info['sfreq'], raw_file.first_samp, axes=axes)
    print('1st event at ', raw_file.times[events[0, 0]])
    print('last event at ', raw_file.times[events[-1, 0]])
    trial_event_times = raw_file.times[events[:, 0]]


def get_eeg_picks(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - the eeg picks
    """
    eeg_picks = mne.pick_types(raw_file.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])
    return eeg_picks


def plot_psd(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - Plots the psd of the raw file
    """
    eeg_picks = get_eeg_picks(raw_file)
    mne.viz.plot_raw_psd(raw_file)
    raw_file.plot_psd(area_mode='range', ax=plt.gca(), picks=eeg_picks, fmax=250)
    plt.figure(figsize=(17, 5))
    raw_file.plot_psd(area_mode='range', ax=plt.gca(), picks=eeg_picks, fmax=35)


def band_pass_filter(raw_file, low_frequency, high_frequency):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function, low/high frequency cut off points
    :return:    - The raw file after applying the filter using the given values
    """
    eeg_picks = get_eeg_picks(raw_file)
    raw_file = raw_file.load_data().filter(low_frequency, high_frequency, picks=eeg_picks, filter_length='auto',
                                l_trans_bandwidth=0.1, h_trans_bandwidth=0.5, method='fft',
                                n_jobs=4, verbose=True)
    return raw_file

def get_eog_events(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - The eog events within the raw data file
    """
    eog_event_id = 5000
    eog_events = mne.preprocessing.find_eog_events(raw_file, eog_event_id)
    return eog_events


def epoch_eog_events(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - The epoched eog data from the raw file
    """
    eog_event_id = 5000
    eog_events = get_eog_events(raw_file)
    picks = mne.pick_types(raw_file.info, meg=False, eeg=True, eog=True, stim=True, exclude=[])
    eog_epochs = mne.Epochs(raw_file, events=eog_events, event_id=eog_event_id,
                            tmin=-.5, tmax=.5, proj=False, picks=picks,
                            preload=True, verbose=False)
    return eog_epochs


def plot_eog(eog_epochs):
    """
    :param:     - eog_epochs, Load in the eog_epochs taken from the epoch_eog_events() function
    :return:    - Plots the epoched eog data
    """
    eog_evoked = eog_epochs.average()
    eog_evoked.plot()


def get_ica(raw_file):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function
    :return:    - Fits ICA to the raw file and then returns the raw file
    """
    n_components = 64  # if float, select n_components by explained variance of PCA
    method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
    decim = 3  # we need sufficient statistics, not all time points -> saves time
    random_state = np.random.RandomState(42)

    # we will also set state of the random number generator - ICA is a
    # non-deterministic algorithm, but we want to have the same decomposition
    # and the same order of components each time this tutorial is run
    random_state = 23

    ica = ICA(n_components=n_components, max_iter='auto', random_state=random_state)
    ica = ica.fit(raw_file)
    return ica

def apply_ica(raw_file, ica):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function: ica, pass in the ica data from the get_ica() function
    :return:    - Applies ICA to the raw file and then returns the raw file
    """
    clean = ica.apply(raw_file)
    clean.plot()
    return clean


def write_to_csv(df,path):
    """
    :param:     - raw_file, pass in the loaded file from the import_data function: path, locaiton to file
    :return:    - Writes the data frame of the cleaned raw data to a csv at given path
    """
    df.to_csv(path)