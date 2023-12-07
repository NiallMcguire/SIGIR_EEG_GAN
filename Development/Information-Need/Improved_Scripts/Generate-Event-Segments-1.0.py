import preprocessing_pipeline as pp
import mne
import numpy as np
import pickle


def load_raw_data(path):
    raw = pp.load_eeg_data(path, eeg_format="cnt", use_montage_creator=True, montage_type='standard_1020')
    raw = pp.add_stimulus_to_raw(raw)
    raw = pp.bandpass_filter(raw)
    raw = pp.average_rereference(raw)
    raw = pp.average_rereference(raw)
    events = mne.find_events(raw, stim_channel='STI')
    ica_tool = pp.ICAChannelSelection(n_component=20, data=raw)  # You may adjust `n_component` based on your needs.
    ica_tool.fit_ica()
    combined_artifacts = ica_tool.find_combined_artifacts()

    # Apply ICA to remove the identified artifact components
    cleaned_data = ica_tool.apply_ica(combined_artifacts)
    cleaned_data = cleaned_data.pick_types(eeg=True)


    return cleaned_data, events

def get_need_to_search_event_times(events):
    #Load in the IDs specific to word displays, as well as NeedToSearch conditional IDs
    SearchIDs = [21,22,23,24,25,26,27,28,29,30,31,32,33,4,5,6,9]

    events_times = mne.pick_events(events,include=SearchIDs)
    events_times = np.where(events_times == 4, 34, events_times) #Change 4 & 5 to 34 & 35 to allow loop to run
    events_times = np.where(events_times == 5, 35, events_times)
    events_times = np.where(events_times == 6, 36, events_times) #Change 6 & 9 to 34 & 35 to allow loop to run
    events_times = np.where(events_times == 9, 37, events_times)

    #Segment all trials into lists of events
    event_sequence_list = []
    event_sequence = []
    event_id_max = 0
    for x in events_times:
        event_info = x
        event_id = event_info[2]
        #print(event_id_max)
        if event_id > event_id_max:
            event_id_max = event_id
            event_sequence.append(event_info)
        else:
            event_sequence_list.append(event_sequence)
            event_sequence = []
            event_id_max = 0

    #Find all trials that contain both 34 and 35 e.g. NeedToSearch conditions
    NeedToSearchEventTimes = []
    CorrectEventTimes = []
    InCorrectEventTimes = []
    for event in event_sequence_list:
        event_sequence = event
        if any(36 and 37 in array for array in event_sequence) == True:
            NeedToSearchEventTimes.append(event_sequence)
        elif any(34 in array for array in event_sequence) == True:
            CorrectEventTimes.append(event_sequence)
        elif any(35 in array for array in event_sequence) == True:
            InCorrectEventTimes.append(event_sequence)

    return NeedToSearchEventTimes, CorrectEventTimes, InCorrectEventTimes

def get_eeg_time_segments(raw, EventTimes):
    raw_copy = raw.copy()
    sample_rate = 500 #TODO - get this from raw
    eeg_segments = []
    egg_event_ids = []
    for event_sequence in EventTimes:
        EEG_event_sequence = []
        EEG_event_ids = []
        for event in event_sequence:
            start_index = event[0]
            end_index = int(start_index + (0.8 * sample_rate))
            #end_index = int(start_index + (0.8*sample_rate))
            cropped_data = raw_copy[:, start_index:end_index]
            EEG_event_sequence.append(cropped_data)
            EEG_event_ids.append(event[2])
        eeg_segments.append(EEG_event_sequence)
        egg_event_ids.append(EEG_event_ids)
    return eeg_segments, egg_event_ids


if __name__ == "__main__":
    Subject_EEG_Segments_ID = {}
    ParticipantList = ["01", "02", "03", "04", "05", "06", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
                       "18", "19", "20", "21", "22", "23", "24"]

    for participant_number in ParticipantList:
        path = fr"I:\Science\CIS-YASHMOSH\niallmcguire\dominika\Raw Data\0{participant_number}\0{participant_number}.cnt"
        raw, events = load_raw_data(path)
        NeedToSearchEventTimes, CorrectEventTimes, InCorrectEventTimes = get_need_to_search_event_times(events)
        NeedToSearchEEGSegments = get_eeg_time_segments(raw, NeedToSearchEventTimes)
        CorrectEventEEGSegments = get_eeg_time_segments(raw, CorrectEventTimes)
        InCorrectEventEEGSegments = get_eeg_time_segments(raw, InCorrectEventTimes)
        Subject_EEG_Segments_ID[participant_number] = [NeedToSearchEEGSegments, CorrectEventEEGSegments, InCorrectEventEEGSegments]
        pickle_file_path = r'C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Information-Need\Data\DataSegments\EEG_Event_Segments.pkl'

        # Open the file in binary write mode and use pickle.dump to save the data
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(Subject_EEG_Segments_ID, pickle_file)


