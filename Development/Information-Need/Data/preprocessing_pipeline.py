import mne
import os
from scipy.stats import zscore
import numpy as np
import pandas as pd
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs


class MontageCreator:
    """
    A utility class to assist in handling montages in MNE-Python.
    It provides functionalities to rename channels to conform to the standard 10-20 system,
    set the standard 10-20 montage, and set a custom montage based on digitized coordinates.

    Attributes:
        raw (mne.io.Raw): The raw EEG data.
        ch_coords (dict): Coordinates extracted from raw data's digitizers.
        mapped_ch_positions (dict): Actual EEG channel names mapped to their coordinates.
    """

    def __init__(self, raw):
        """
        Initialize MontageCreator with raw EEG data.

        Args:
            raw (mne.io.Raw): The raw EEG data.
        """
        self.raw = raw
        self.ch_coords = self.extract_dig_coords()
        self.mapped_ch_positions = self.map_coords_to_ch_names(self.ch_coords)

    def rename_channels_to_1020(self):
        """
        Rename EEG channels to conform to the standard 10-20 system.
        """
        rename_dict = {
            'FP1': 'Fp1', 'FP2': 'Fp2',
            'FZ': 'Fz', 'FCZ': 'FCz',
            'CZ': 'Cz', 'CPZ': 'CPz',
            'PZ': 'Pz', 'OZ': 'Oz'
        }
        self.raw.rename_channels(rename_dict)

    def set_1020_montage(self):
        """
        Rename EEG channels to the standard 10-20 system (if necessary)
        and set the standard 10-20 montage to the raw EEG data.
        """
        # This ensures the channels are correctly named before setting the montage
        self.rename_channels_to_1020()

        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(montage)

    def extract_dig_coords(self):
        """
        Extract x, y, z coordinates from the raw data's digitizers.

        Returns:
            dict: Dictionary mapping channel names (like 'EEG #1') to their (x, y, z) coordinates.
        """
        coords = {}
        for point in self.raw.info['dig']:
            if point['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG:
                channel_name = f"EEG #{point['ident']}"
                coords[channel_name] = (point['r'][0], point['r'][1], point['r'][2])
        return coords

    def map_coords_to_ch_names(self, ch_coords):
        """
        Map the 'EEG #x' coordinates to the actual channel names.

        Args:
            ch_coords (dict): Dictionary with 'EEG #x' as keys and their coordinates as values.

        Returns:
            dict: Dictionary with actual EEG channel names as keys and their coordinates as values.
        """
        raw_channels = self.raw.info['ch_names']
        if len(raw_channels) != len(ch_coords):
            raise ValueError("Mismatch between number of channels and number of coordinates.")
        eeg_channel_mapping = {f"EEG #{i + 1}": name for i, name in enumerate(raw_channels)}
        mapped_coords = {eeg_channel_mapping[key]: value for key, value in ch_coords.items()}
        return mapped_coords

    def set_custom_montage(self):
        """
        Create and set a custom montage for the raw EEG data based on the provided channel coordinates.
        """
        new_montage = mne.channels.make_dig_montage(
            ch_pos=self.mapped_ch_positions,
            coord_frame='head'
        )
        self.raw.set_montage(new_montage)


# Usage:
# raw = ...  # Load your raw data here
# montage_creator = MontageCreator(raw)
# montage_creator.rename_channels_to_1020()  # Optional: Only if you want to rename channels
# montage_creator.set_1020_montage()  # Optional: Only if you want the standard 10-20 montage
# montage_creator.set_custom_montage()  # Optional: Only if you want a custom montage


def load_eeg_data(file, channelloc=None, preload=True, eeg_format=None, channel_format=None,
                  standard_montage_name=None, use_montage_creator=False, montage_type='standard_1020'):
    """
    Load raw EEG data and, if provided, apply channel locations.

    Args:
        file (str): Path to the raw EEG data file.
        channelloc (str, optional): Path to the channel locations file.
        preload (bool, optional): Load the raw data into memory. Defaults to True.
        eeg_format (str, optional): Format of the raw EEG data file.
            E.g., 'cnt', 'fif', 'edf', 'bdf', 'set', 'vhdr'.
        channel_format (str, optional): Format of the channel locations file.
            E.g., 'elp', 'sfp', 'hpts'.
        standard_montage_name (str, optional): Name of a standard montage to use
            if no custom channel file is provided.
        use_montage_creator (bool, optional): If True, use a montage creator tool. Defaults to False.
        montage_type (str, optional): Either 'standard_1020' or 'custom'. Defaults to 'standard_1020'.

    Returns:
        mne.io.Raw: EEG data with channel locations applied (if provided).
    """
    # Check if EEG data file exists
    if not os.path.exists(file):
        raise FileNotFoundError(f"No such file or directory: '{file}'")
    # Check if channel locations file exists (if provided)
    if channelloc is not None and not os.path.exists(channelloc):
        raise FileNotFoundError(f"No such file or directory: '{channelloc}'")

    # Load raw EEG data based on the provided file format
    if eeg_format is None:
        raise ValueError("Please specify the EEG format.")
    elif eeg_format == 'cnt':
        raw = mne.io.read_raw_cnt(file, preload=preload)
    elif eeg_format == 'fif':
        raw = mne.io.read_raw_fif(file, preload=preload)
    elif eeg_format == 'edf':
        raw = mne.io.read_raw_edf(file, preload=preload)
    elif eeg_format == 'bdf':
        raw = mne.io.read_raw_bdf(file, preload=preload)
    elif eeg_format == 'set':
        raw = mne.io.read_raw_eeglab(file, preload=preload)
    elif eeg_format == 'vhdr':
        raw = mne.io.read_raw_brainvision(file, preload=preload)
    else:
        raise ValueError(f"Unsupported data format: '{eeg_format}'")

    # Set specific channels as 'eog'
    # This step assigns the 'eog' type to a predefined set of channels based on their names.
    # NOTE: This function assumes that the EEG data uses these specific channel names for EOG channels.
    # If the EEG data has different names for EOG channels, you'll need to update this dictionary accordingly.
    channel_types = {
        'HEOL': 'eog',  # Left horizontal EOG
        'HEOR': 'eog',  # Right horizontal EOG
        'VEOU': 'eog',  # Upper vertical EOG
        'VEOL': 'eog',  # Lower vertical EOG
        #         'A1': 'eog',    # Auxiliary EOG (commonly used for mastoid reference or other auxiliary channels)
        #         'A2': 'eog'     # Auxiliary EOG (commonly used for mastoid reference or other auxiliary channels)
    }
    raw.set_channel_types(channel_types)

    # raw.set_eeg_reference(ref_channels=['A1', 'A2'])
    # Explicitly set A1 and A2 as misc (miscellaneous) type if you do not want them to be treated as EEG channels
    raw.set_channel_types({'A1': 'misc', 'A2': 'misc'})

    # Load channel locations, either from a custom file or a standard montage
    if channelloc is not None:
        if channel_format is None:
            raise ValueError("Please provide a channel format or use the standard montage name.")
        elif channel_format in ['elp', 'sfp', 'hpts']:
            montage = mne.channels.read_custom_montage(channelloc)
        else:
            raise ValueError(f"Unsupported channel format: '{channel_format}'")
    elif standard_montage_name is not None:
        montage = mne.channels.make_standard_montage(standard_montage_name)
    else:
        montage = None

    # If a montage is loaded, apply it to the raw data
    if montage is not None:
        raw.set_montage(montage)
    # Or use the montage creator tool if specified  (i.e class MontageCreator)
    elif use_montage_creator:
        montage_handler = MontageCreator(raw)
        if montage_type == 'standard_1020':
            montage_handler.set_1020_montage()
        elif montage_type == 'custom':
            montage_handler.set_custom_montage()
        else:
            raise ValueError(f"Invalid montage_type: {montage_type}. Expected 'standard_1020' or 'custom'.")

    return raw


# Example usage:
# raw = load_eeg_data(
#     file=eeg_file_path,
#     channelloc=channel_loc_path,
#     preload=True,
#     eeg_format=eeg_format,
#     channel_format=channel_format,
#     standard_montage_name=None,
#     use_montage_creator=False,
#     montage_type='standard_1020'
# )


def add_stimulus_to_raw(raw):
    # Create an events array from the annotations
    event_ids = raw.annotations.description

    # Convert to a NumPy array of integers
    event_ids = event_ids.astype(int)

    # Create an empty stimulus channel
    stim_data = np.zeros((1, len(raw.times)))

    # Loop through your annotations
    for annot in raw.annotations:
        # Check if the event ID is in your list of IDs
        if int(annot["description"]) in event_ids:
            # Find the sample corresponding to the annotation onset
            sample = int(raw.time_as_index(annot["onset"]))

            # Add a marker to the stimulus channel
            stim_data[0, sample] = int(annot["description"])

    # Add the stimulus channel to the Raw object
    info = mne.create_info(["STI"], raw.info["sfreq"], ch_types=["stim"])
    stim_raw = mne.io.RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)

    return raw