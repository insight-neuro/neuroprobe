import h5py
import os
import json
import pandas as pd
import numpy as np
import torch

from .config import *

class BrainTreebankSubject:
    """ 
        This class is used to load the neural data for a given subject and trial.
        It also contains methods to get the data for a given electrode and trial, and to get the spectrogram for a given electrode and trial.
    """
    def __init__(self, subject_id, allow_corrupted=False, allow_missing_coordinates=False, cache=False, dtype=torch.float32, coordinates_type="cortical"):
        self.subject_id = subject_id
        self.subject_identifier = f'btbank{subject_id}'
        self.allow_corrupted = allow_corrupted
        self.allow_missing_coordinates = allow_missing_coordinates
        self.cache = cache
        self.dtype = dtype  # Store dtype as instance variable
        self.coordinates_type = coordinates_type
        assert coordinates_type in ["cortical", "mni", "lpi"], "Invalid coordinates type. Must be one of: 'cortical', 'mni', 'lpi'"

        self.localization_data = self._load_localization_data()
        self.electrode_labels = self._get_all_electrode_names()

        self.h5_neural_data_keys = {e:"electrode_"+str(i) for i, e in enumerate(self.electrode_labels)} # only used for accessing the neural data in h5 files
        self.electrode_labels = self._filter_electrode_labels()
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}

        self.electrode_data_length = {}
        self.neural_data_cache = {} # structure: {trial_id: torch.Tensor of shape (n_electrodes, n_samples)}
        self.h5_files = {} # structure: {trial_id: h5py.File}

        self.neural_data_cache_window_from = 0
        self.neural_data_cache_window_to = None

    def set_electrode_subset(self, electrode_labels):
        """
            Set the subset of electrodes to use for this subject.
        """
        self.electrode_labels = electrode_labels
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}
        if self.cache:
            self.clear_neural_data_cache() # clear the cache to avoid using the wrong electrodes

    def get_n_electrodes(self):
        return len(self.electrode_labels)
    def _load_localization_data(self):
        """Load localization data for this electrode's subject from depth-wm.csv"""
        loc_file = os.path.join(ROOT_DIR, f'localization/sub_{self.subject_id}/depth-wm.csv')
        df = pd.read_csv(loc_file)
        df['Electrode'] = df['Electrode'].apply(self._clean_electrode_label)
        return df
    def _get_all_electrode_names(self):
        electrode_labels_file = os.path.join(ROOT_DIR, f'electrode_labels/sub_{self.subject_id}/electrode_labels.json')
        electrode_labels = json.load(open(electrode_labels_file))
        electrode_labels = [self._clean_electrode_label(e) for e in electrode_labels]
        return electrode_labels
    def _clean_electrode_label(self, electrode_label):
        return electrode_label.replace('*', '').replace('#', '')
    def _get_corrupted_electrodes(self):
        corrupted_electrodes_file = os.path.join(ROOT_DIR, "corrupted_elec.json")
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[f'sub_{self.subject_id}']]

        if not self.allow_missing_coordinates:
            missing_coordinate_electrodes = {
                "sub_1": ["F3cId10"], "sub_2": [], 
                "sub_3": ["F3c9", "F3c10", "T1aIc1", "T1aIc2", "P2a10", "O1aIb2", "O1aIb3", "O1aIb4", "O1aIb5", "O1aIb6", "O1aIb7", "O1aIb8"], 
                "sub_4": ["LT1aIb10", "LF3bIa12"], "sub_5": [], "sub_6": [], "sub_7": ["LF3aOFa16", "LF1cCb12"], "sub_8": ["F2bCb6", "F2bCb14"], 
                "sub_9": ["P2a6", "P2a7", "P2a8"], "sub_10": ["T1aIa4", "P2cCc5"]}[f'sub_{self.subject_id}']
            corrupted_electrodes += missing_coordinate_electrodes
        return list(set(corrupted_electrodes))
    def _filter_electrode_labels(self):
        """
            Filter the electrode labels to remove corrupted electrodes and electrodes that don't have brain signal
        """
        filtered_electrode_labels = self.electrode_labels
        # Step 1. Remove corrupted electrodes
        if not self.allow_corrupted:
            corrupted_electrodes = self._get_corrupted_electrodes()
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Step 2. Remove trigger electrodes
        trigger_electrodes = [e for e in self.electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG"))]
        filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in trigger_electrodes]
        return filtered_electrode_labels
    
    def cache_neural_data(self, trial_id, cache_window_from=None, cache_window_to=None, force_cache=False):
        assert self.cache, "Cache is not enabled; not able to cache neural data."

        # Open file with context manager to ensure proper closing
        neural_data_file = os.path.join(ROOT_DIR, f'sub_{self.subject_id}_trial{trial_id:03}.h5')
        with h5py.File(neural_data_file, 'r') as f:
            # Get data length first
            self.electrode_data_length[trial_id] = f['data'][self.h5_neural_data_keys[self.electrode_labels[0]]].shape[0]
            if cache_window_to is None: cache_window_to = self.electrode_data_length[trial_id]
            
            if cache_window_from is None: cache_window_from = 0
            if (trial_id in self.neural_data_cache) and \
                (not force_cache) and \
                ((self.cache_neural_data_window_from is None) or (self.cache_neural_data_window_from <= cache_window_from)) and \
                ((self.cache_neural_data_window_to is None) or (self.cache_neural_data_window_to >= cache_window_to)): 
                return  # no need to cache again

            self.cache_neural_data_window_from = cache_window_from
            self.cache_neural_data_window_to = cache_window_to
            
            # Pre-allocate tensor with specific dtype
            self.neural_data_cache[trial_id] = torch.zeros((len(self.electrode_labels), cache_window_to-cache_window_from), dtype=self.dtype)

            # Load data
            for electrode_label, electrode_id in self.electrode_ids.items():
                neural_data_key = self.h5_neural_data_keys[electrode_label]
                self.neural_data_cache[trial_id][electrode_id] = torch.from_numpy(f['data'][neural_data_key][cache_window_from:cache_window_to]).to(self.dtype)

    def _get_all_laplacian_electrodes(self, verbose=False):
        """
            Get all laplacian electrodes for a given subject. This function is originally from
            https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)
        """
        def stem_electrode_name(name):
            #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
            #names look like 'T1b2
            reverse_name = reversed(name)
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return ''.join(reversed(stem)), int(''.join(reversed(num)))
        def has_neighbors(stem, stems):
            (x,y) = stem
            return ((x,y+1) in stems) and ((x,y-1) in stems)
        def get_neighbors(stem):
            (x,y) = stem
            return [f'{x}{y}' for (x,y) in [(x,y+1), (x,y-1)]]
        stems = [stem_electrode_name(e) for e in self.electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
        neighbors = {e: get_neighbors(stem_electrode_name(e)) for e in electrodes}
        return electrodes, neighbors

    def clear_neural_data_cache(self, trial_id=None):
        if trial_id is None:
            self.neural_data_cache = {}
            self.h5_files = {}
        else:
            if trial_id in self.neural_data_cache: del self.neural_data_cache[trial_id]
            if trial_id in self.h5_files: del self.h5_files[trial_id]
        self.electrode_data_length = {}
        self.neural_data_cache_window_from = 0
        self.neural_data_cache_window_to = None
    def open_neural_data_file(self, trial_id):
        assert not self.cache, "Cache is enabled; Use cache_neural_data() instead."
        if trial_id in self.h5_files: return
        neural_data_file = os.path.join(ROOT_DIR, f'sub_{self.subject_id}_trial{trial_id:03}.h5')
        self.h5_files[trial_id] = h5py.File(neural_data_file, 'r')
        self.electrode_data_length[trial_id] = self.h5_files[trial_id]['data'][self.h5_neural_data_keys[self.electrode_labels[0]]].shape[0]
    def load_neural_data(self, trial_id, cache_window_from=None, cache_window_to=None):
        if self.cache: self.cache_neural_data(trial_id, cache_window_from=cache_window_from, cache_window_to=cache_window_to)
        else: self.open_neural_data_file(trial_id)
    
    def get_electrode_coordinates(self):
        if self.coordinates_type == "cortical":
            return self.get_electrode_coordinates_cortical()
        elif self.coordinates_type == "mni":
            return self.get_electrode_coordinates_mni()
        elif self.coordinates_type == "lpi":
            return self.get_electrode_coordinates_lpi()
        else:
            raise ValueError(f"Invalid coordinates type: {self.coordinates_type}")
    def get_electrode_coordinates_cortical(self):
        """
            Get the coordinates of the electrodes for this subject
            Returns:
                coordinates: (n_electrodes, 3) tensor of STANDARDIZED BRAIN ATLAS CORTICAL PROJECTION OF THE coordinates (X, Y, Z) in mm
        """
        loc_file = os.path.join(ROOT_DIR, f'localization/elec_coords_full.csv')
        df = pd.read_csv(loc_file)

        # Remove asterisks from electrode names in the dataframe
        df['Electrode'] = df['Electrode'].str.replace('*', '', regex=False)
        
        electrode_coordinates = torch.zeros((len(self.electrode_labels), 3), dtype=self.dtype) * np.nan
        for electrode_label in self.electrode_labels:
            # Find the row for this electrode and subject
            electrode_row = df[(df['Electrode'] == electrode_label) & (df['Subject'] == f'sub_{self.subject_id}')]
            
            if not electrode_row.empty:
                electrode_id = self.electrode_ids[electrode_label]
                electrode_coordinates[electrode_id] = torch.tensor([
                    electrode_row.iloc[0]['X'],
                    electrode_row.iloc[0]['Y'], 
                    electrode_row.iloc[0]['Z']
                ], dtype=self.dtype)
        return electrode_coordinates
    def get_electrode_coordinates_mni(self):
        raise NotImplementedError("Direct MNI coordinates are not yet available for the Braintreebank dataset. Will be added in the future ASAP!")
    def get_electrode_coordinates_lpi(self):
        """
            Get the coordinates of the electrodes for this subject
            Returns:
                coordinates: (n_electrodes, 3) tensor of coordinates (L, P, I) without any preprocessing of the coordinates
                All coordinates are in between 50mm and 200mm for this dataset
        """
        # Create tensor of coordinates in same order as electrode_labels
        coordinates = torch.zeros((len(self.electrode_labels), 3), dtype=self.dtype)
        for i, label in enumerate(self.electrode_labels):
            row = self.get_electrode_metadata(label)
            coordinates[i] = torch.tensor([row['L'], row['P'], row['I']], dtype=self.dtype)
        return coordinates

    def get_electrode_metadata(self, electrode_label):
        """
            Get the metadata for a given electrode.
        """
        return self.localization_data[self.localization_data['Electrode'] == electrode_label].iloc[0]
    def get_all_electrode_metadata(self):
        filtered_df = self.localization_data[self.localization_data['Electrode'].isin(self.electrode_labels)]
        ordered_df = pd.DataFrame([filtered_df[filtered_df['Electrode'] == label].iloc[0] for label in self.electrode_labels])
        return ordered_df.reset_index(drop=True)

    def get_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache:
            if trial_id not in self.neural_data_cache: self.cache_neural_data(trial_id)

            # in case we cached a subset of the data, we need to force a re-cache if we want to access data outside of that range
            if ((self.cache_neural_data_window_from is not None) and (window_from < self.cache_neural_data_window_from)) or \
                ((self.cache_neural_data_window_to is not None) and (window_to > self.cache_neural_data_window_to)):
                self.cache_neural_data(trial_id, force_cache=True)

            electrode_id = self.electrode_ids[electrode_label]
            data = self.neural_data_cache[trial_id][electrode_id][window_from-self.cache_neural_data_window_from:window_to-self.cache_neural_data_window_from]
            return data
        else:
            if trial_id not in self.h5_files: self.open_neural_data_file(trial_id)
            neural_data_key = self.h5_neural_data_keys[electrode_label]
            data = torch.from_numpy(self.h5_files[trial_id]['data'][neural_data_key][window_from:window_to]).to(self.dtype)
            return data

    def get_all_electrode_data(self, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache: 
            # in case we cached a subset of the data, we need to force a re-cache if we want to access data outside of that range
            if ((self.cache_neural_data_window_from is not None) and (window_from < self.cache_neural_data_window_from)) or \
                ((self.cache_neural_data_window_to is not None) and (window_to > self.cache_neural_data_window_to)):
                self.cache_neural_data(trial_id, force_cache=True)

            return self.neural_data_cache[trial_id][:, window_from-self.cache_neural_data_window_from:window_to-self.cache_neural_data_window_from]
        else:
            all_electrode_data = torch.zeros((len(self.electrode_labels), window_to-window_from), dtype=self.dtype)
            for electrode_label, electrode_id in self.electrode_ids.items():
                all_electrode_data[electrode_id] = self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to)
            return all_electrode_data

    def get_lite_electrodes(self):
        """
        Return the list of 'lite' electrodes for this subject, as defined in neuroprobe_config.py.
        """
        return NEUROPROBE_LITE_ELECTRODES[f"btbank{self.subject_id}"]
Subject = BrainTreebankSubject # for backwards compatibility


if __name__ == "__main__":
    subject_id, trial_id = 1, 1
    subject = BrainTreebankSubject(subject_id, cache=True, dtype=torch.bfloat16)
    print(subject.get_lite_electrodes())
    subject.load_neural_data(trial_id)
    print(subject.get_all_electrode_data(trial_id).shape)
    exit()