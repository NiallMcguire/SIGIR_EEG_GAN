import pickle
import pandas as pd
import numpy as np

def load_in_participant_data(path, subject):
    with open(path, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)

    loaded_data = loaded_data[subject]

    NeedToSearch = loaded_data[0]
    CorrectSearch = loaded_data[1]
    IncorrectSearch = loaded_data[2]

    return NeedToSearch, CorrectSearch, IncorrectSearch


import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch


def extract_features_extended(eeg_segment):
    features = []

    # Basic statistical features for each channel
    mean_vals = np.mean(eeg_segment, axis=1)
    std_devs = np.std(eeg_segment, axis=1)
    max_vals = np.max(eeg_segment, axis=1)
    min_vals = np.min(eeg_segment, axis=1)
    skewness_vals = skew(eeg_segment, axis=1)
    kurtosis_vals = kurtosis(eeg_segment, axis=1)

    features.extend(mean_vals)
    features.extend(std_devs)
    features.extend(max_vals)
    features.extend(min_vals)
    features.extend(skewness_vals)
    features.extend(kurtosis_vals)

    # Frequency domain features using Welch's method
    for channel_data in eeg_segment:
        f, psd = welch(channel_data, fs=500, nperseg=256)
        psd_bands = [np.sum(psd[(f >= low) & (f < high)]) for (low, high) in [(0, 4), (4, 8), (8, 12), (12, 30), (30, 50)]]
        spectral_entropy = -np.sum(psd * np.log2(psd + 1e-12))  # Avoid log(0)
        peak_frequency = f[np.argmax(psd)]

        features.extend(psd_bands)
        features.append(spectral_entropy)
        features.append(peak_frequency)

    # Time-domain features
    for channel_data in eeg_segment:
        hjorth_params = [np.var(channel_data), np.var(np.diff(channel_data)), np.var(np.diff(np.diff(channel_data)))]
        zero_cross_rate = np.sum(np.diff(np.sign(channel_data)) != 0) / len(channel_data)
        waveform_length = np.sum(np.abs(np.diff(channel_data)))

        features.extend(hjorth_params)
        features.append(zero_cross_rate)
        features.append(waveform_length)

    return np.array(features)


def segments_to_features(Event):
    Event_Queries = Event[0]
    Event_IDs = Event[1]

    EEG_Query_Features = []
    for sentences in range(len(Event_Queries)):
        EEG_sentence_features = []
        for words in range(len(Event_Queries[sentences])):
            #EEG_segment_ID = Event_IDs[sentences][words]
            EEG_segments = Event_Queries[sentences][words][0]
            features = extract_features_extended(EEG_segments)
            EEG_sentence_features.append(features)
        EEG_Query_Features.append(EEG_sentence_features)

    return EEG_Query_Features, Event_IDs



if __name__ == "__main__":
    ParticipantList = ["01", "02", "03", "04", "05", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                       "19", "20", "21", "22", "23", "24"]
    path = fr"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Information-Need\Data\DataSegments\EEG_Event_Segments.pkl"

    ParticipantFeatureDict = {}
    for subject in ParticipantList:
        print(f"Processing Participant {subject}")
        NeedToSearch, CorrectSearch, IncorrectSearch = load_in_participant_data(path, subject)
        NeedToSearchFeatures = segments_to_features(NeedToSearch)
        CorrectSearchFeatures = segments_to_features(CorrectSearch)
        IncorrectSearchFeatures = segments_to_features(IncorrectSearch)

        ParticipantFeatureDict[subject] = [NeedToSearchFeatures, CorrectSearchFeatures, IncorrectSearchFeatures]

    pickle_file_path = fr'C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Information-Need\Data\stat_features\Participant_Features.pkl'

    # Open the file in binary write mode and use pickle.dump to save the data
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(ParticipantFeatureDict, pickle_file)
