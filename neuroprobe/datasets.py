import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os, json
from sklearn import preprocessing
from .config import *
from .braintreebank_subject import BrainTreebankSubject

NEW_FEATURES_FILE_NAME = "test_new_features.csv"

# Defining the names of evaluations and preparing them for downstream processing
single_float_variables_name_remapping = {
    "pitch": "enhanced_pitch", #"pitch",
    "volume": "rms", #"rms",
    "frame_brightness": "mean_pixel_brightness",
    "delta_frame_brightness": "delta_mean_pixel_brightness",
    "delta_frame_brightness_abs": "delta_mean_pixel_brightness_abs",
    "delta_delta_frame_brightness": "delta_delta_mean_pixel_brightness",
    "delta_delta_frame_brightness_abs": "delta_delta_mean_pixel_brightness_abs",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "delta_volume": "delta_rms",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length"
}
classification_variables_name_remapping = {
    "word_head_pos": "bin_head",
    "word_part_speech": "pos"
}
new_pitch_variables = ['enhanced_pitch', 'enhanced_volume', 'delta_enhanced_pitch', 'delta_enhanced_volume', 'raw_pitch', 'raw_volume', 'delta_raw_pitch', 'delta_raw_volume']
single_float_variables = list(single_float_variables_name_remapping.values()) + list(single_float_variables_name_remapping.keys()) + new_pitch_variables
classification_variables = list(classification_variables_name_remapping.values()) + list(classification_variables_name_remapping.keys())
all_tasks = single_float_variables + ["onset", "speech"] + ["face_num", "word_gap", "word_index"] + classification_variables + ["scene", "scene_onset", "speaker", "speaker_gender"] + ["delta_face_num", "delta_face_num_abs", "delta_delta_face_num", "delta_delta_face_num_abs"]


speaker_gender = {
    'Leland Turbo': 'male',
    'Acer': 'male',
    'Crabby': 'male',
    'Finn McMissile': 'male',
    'combat ship': 'unknown',
    'Tannoy': 'unknown',
    'Professor Zündapp': 'male',
    'Grem': 'male',
    'Tony': 'male',
    'Mater': 'male',
    'Otis': 'male',
    'Fio': 'female',
    'Filmore': 'male',
    'Sarge': 'male',
    'Lightning McQueen': 'male',
    'Ramone': 'male',
    'Sally': 'female',
    '* not in audio': 'unknown',
    'Mel Dorado': 'male',
    'Miles Axlerod': 'male',
    'Francesco Bernoulli': 'male',
    'caller': 'unknown',
    'Guido': 'male',
    'Luigi': 'male',
    'Lewis Hamilton': 'male',
    'Jeff Gorvette': 'male',
    'announcer': 'unknown',
    'Holly Shiftwell': 'female',
    'sushi chef': 'unknown',
    'Rod Redline': 'male',
    '* multiple speakers': 'unknown',
    'Brent Mustangburger': 'male',
    'David Hobbscap': 'male',
    'Darrell Cartrip': 'male',
    '(various cars in the pit)': 'unknown',
    'Acer * Grem': 'male',
    '*not in audio': 'unknown',
    'reporter #1': 'unknown',
    'reporter #2': 'unknown',
    'Siddeley': 'male',
    'Mater (letter)': 'male',
    'Lightning McQueen (reading letter)': 'male',
    'Luigi (reading)': 'male',
    'Mat': 'unknown',
    'Ma': 'female',
    'french car': 'unknown',
    'Tomber': 'male',
    'Holly Shiftwell * Finn MicMissile': 'mixed',
    'Uncle Topolino': 'male',
    'Aunt Topolino': 'female',
    'Stephenson': 'male',
    'computer': 'unknown',
    'italian track announcer': 'male',
    'Alexander Hugo': 'male',
    'Ivan the towtruck': 'male',
    'Victor Hugo': 'male',
    'casino employee': 'unknown',
    'gambler': 'unknown',
    'serving car': 'unknown',
    'J. Curby Gremlin': 'male',
    'Vladimir Trunkov': 'male',
    'Tubbs Pacer': 'male',
    'disguised voice': 'unknown',
    'lemon': 'male',
    'lemon ': 'male',
    'police car #1': 'unknown',
    'entourage #1': 'unknown',
    'entourage #2': 'unknown',
    'reporter car * Mater': 'mixed',
    'police car #2': 'unknown',
    'reporter car': 'unknown',
    'encourage #2': 'unknown',
    'encourage #1': 'unknown',
    'Grem * Acer': 'male',
    'Flo': 'female',
    'british police car': 'unknown',
    'goon car': 'unknown',
    'Sheriff': 'male',
    'british corporal': 'male',
    'queen car': 'female',
    'secret service car': 'unknown',
    'royal grandson car': 'male',
    'royal presenter': 'unknown',
    'Minny': 'female',
    'Van': 'male',
    'Lizzie': 'female',
    'Mack': 'male',
    'Lightning McQueen * Francesco Bernoulli': 'male',
}



class BrainTreebankSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject, trial_id, dtype, eval_name, output_indices=False, binary_tasks=True,
                 start_neural_data_before_word_onset=START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE, end_neural_data_after_word_onset=END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE,
                 lite=True, nano=False, random_seed=NEUROPROBE_GLOBAL_RANDOM_SEED, output_dict=True, max_samples=None, always_cache_full_subject=False):
        """
        Args:
            subject (Subject): the subject to evaluate on
            trial_id (int): the trial to evaluate on
            dtype (torch.dtype): the data type of the returned data
            eval_name (str): the name of the variable to evaluate on
                Options for eval_name (from the Neuroprobe paper):
                    frame_brightness, global_flow, local_flow, face_num, volume, pitch, delta_volume, 
                    speech, onset, scene_onset, gpt2_surprisal, word_length, word_gap, word_index, word_head_pos, word_part_speech
            lite (bool, optional): if True, the eval is Neuroprobe (the default), otherwise it is Neuroprobe-Full
            nano (bool, optional): if True, the eval is Neuroprobe-Nano, otherwise it is Neuroprobe-Lite (if lite is True - this is the default)

            output_indices (bool, optional): 
                if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
                if False, the dataset will output the neural data directly

            binary_tasks (bool, optional):
                if True, the tasks will all be binary (default).
                if False, the tasks will be multi-class classification with a variable number of classes, as described in the technical paper.

            output_dict (bool, optional): 
                if True, the dataset will output a dictionary with the following keys:
                    "data": the neural data -- either directly or as a tuple (index_from, index_to)
                    "label": the label
                    "electrode_labels": the labels of the electrodes
                If False, the dataset will output a tuple (input, label) or ((index_from, index_to), label) directly
            
            start_neural_data_before_word_onset (int, optional): the number of samples to start the neural data before each word onset (defaults to START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE)
            end_neural_data_after_word_onset (int, optional): the number of samples to end the neural data after each word onset (defaults to END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)
            random_seed (int, optional): seed for random operations within this dataset (defaults to NEUROPROBE_GLOBAL_RANDOM_SEED)
            max_samples (int, optional): the maximum number of samples to include in the dataset (defaults to None, which means default limits: none for Neuroprobe-Full, 3500 for Neuroprobe-Lite, 1000 for Neuroprobe-Nano)
            always_cache_full_subject (bool, optional): if True, the dataset will always cache the full subject's neural data (defaults to False)
        """

        # Set up a local random state with the provided seed
        self.rng = np.random.RandomState(random_seed)
        
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"

        self.subject = subject
        self.subject_id = subject.subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.dtype = dtype
        self.binary_tasks = binary_tasks
        self.output_indices = output_indices
        self.start_neural_data_before_word_onset = start_neural_data_before_word_onset
        self.end_neural_data_after_word_onset = end_neural_data_after_word_onset
        self.lite = lite
        self.nano = nano
        self.output_dict = output_dict
        self.max_samples = max_samples
        self.always_cache_full_subject = always_cache_full_subject

        if self.nano:
            nano_electrodes = NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in nano_electrodes if e in subject.electrode_labels]
            self.electrode_labels = [subject.electrode_labels[i] for i in self.electrode_indices_subset]
            subject_trial = (subject.subject_id, self.trial_id)
            assert subject_trial in NEUROPROBE_NANO_SUBJECT_TRIALS, f"Subject {subject.subject_id} trial {self.trial_id} not in NEUROPROBE_NANO_SUBJECT_TRIALS"
        elif self.lite:
            lite_electrodes = NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in lite_electrodes if e in subject.electrode_labels]
            self.electrode_labels = [subject.electrode_labels[i] for i in self.electrode_indices_subset]
            subject_trial = (subject.subject_id, self.trial_id)
            assert subject_trial in NEUROPROBE_LITE_SUBJECT_TRIALS, f"Subject {subject.subject_id} trial {self.trial_id} not in NEUROPROBE_LITE_SUBJECT_TRIALS"
        else:
            # use all electrode labels and indices
            self.electrode_indices_subset = np.arange(len(subject.electrode_labels))
            self.electrode_labels = subject.electrode_labels
        self.electrode_coordinates = subject.get_electrode_coordinates()[self.electrode_indices_subset]

        eval_name_remapped = eval_name
        if eval_name in single_float_variables_name_remapping: eval_name_remapped = single_float_variables_name_remapping[eval_name]
        if eval_name in classification_variables_name_remapping: eval_name_remapped = classification_variables_name_remapping[eval_name]
        self.eval_name_remapped = eval_name_remapped

        words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_words_df.csv")
        nonverbal_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_nonverbal_df.csv")
        self.all_words_df = pd.read_csv(words_df_path)
        self.nonverbal_df = pd.read_csv(nonverbal_df_path)

        self.movie_name = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"{self.subject.subject_identifier}_{self.trial_id}"]
        
        # Add the original features from braintreebank to the all_words_df
        # transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{self.movie_name}/features.csv')
        # transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{self.movie_name}/test_new_delta_pixel.csv')
        # transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{self.movie_name}/test_new_delta_pixel_and_face.csv')
        transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{self.movie_name}/{NEW_FEATURES_FILE_NAME}')
        original_features_df = pd.read_csv(transcript_file_format.format(self.movie_name)).set_index('Unnamed: 0')
        # Add new columns from words_df using original_index mapping
        new_columns = [col for col in original_features_df.columns if col not in self.all_words_df.columns]
        for col in new_columns:
            self.all_words_df[col] = self.all_words_df['original_index'].map(original_features_df[col])
        
        if eval_name in single_float_variables:
            # Grab the new pitch volume features if they exist
            if self.eval_name_remapped in new_pitch_variables:
                pitch_volume_features_path = os.path.join(PITCH_VOLUME_FEATURES_DIR, f"{self.movie_name}_pitch_volume_features.json")
                with open(pitch_volume_features_path, 'r') as f:
                    raw_pitch_volume_features = json.load(f)

                TARGET_DP_FOR_KEYS = 5  # Standard number of decimal places
                normalized_pvf = {}
                for k_str, v_val in raw_pitch_volume_features.items():
                    k_float = float(k_str)
                    normalized_key = f"{k_float:.{TARGET_DP_FOR_KEYS}f}"
                    normalized_pvf[normalized_key] = v_val
                pitch_volume_features = normalized_pvf

                start_times = self.all_words_df['start'].to_list()
                all_labels = []
                for start_time_val in start_times:
                    lookup_key = f"{start_time_val:.{TARGET_DP_FOR_KEYS}f}"
                    label = pitch_volume_features[lookup_key][self.eval_name_remapped]
                    all_labels.append(label)
                all_labels = np.array(all_labels)
            else:
                all_labels = self.all_words_df[self.eval_name_remapped].to_numpy()

            # Get indices for words in top and bottom quartiles
            label_percentiles = np.array([np.mean(all_labels < x) for x in all_labels])
            if self.binary_tasks:
                self.label_indices = {
                    1: np.where(label_percentiles > 0.75)[0],
                    0: np.where(label_percentiles < 0.25)[0]
                }
            else:
                self.label_indices = {
                    2: np.where(label_percentiles >= 0.75)[0],
                    1: np.where((label_percentiles < 0.625) & (label_percentiles >= 0.375))[0],
                    0: np.where(label_percentiles < 0.25)[0]
                }
        elif eval_name in ["onset", "speech"]:
            if eval_name == "onset":
                self.label_indices = {
                    1: np.where(self.all_words_df["is_onset"].to_numpy() == 1)[0], # positive indices
                    0: np.arange(len(self.nonverbal_df)) # negative indices
                }
            elif eval_name == "speech":
                self.label_indices = {
                    1: np.arange(len(self.all_words_df)), # positive indices
                    0: np.arange(len(self.nonverbal_df)) # negative indices
                }
            # elif eval_name == "scene_onset":
            #     # Detect scene changes: find words where the scene label changes from the previous word
            #     # Try common column names for scene labels
            #     scene_col = None
            #     for col_name in ["scene", "scene_id", "scene_label", "scene_number"]:
            #         if col_name in self.all_words_df.columns:
            #             scene_col = col_name
            #             break
                
            #     if scene_col is None:
            #         raise ValueError(f"Scene column not found in features. Expected one of: scene, scene_id, scene_label, scene_number")
                
            #     scene_values = self.all_words_df[scene_col].to_numpy()
            #     # Detect scene changes: scene is different from previous word (or first word is always a scene onset)
            #     is_scene_onset = np.zeros(len(self.all_words_df), dtype=bool)
            #     is_scene_onset[0] = True  # First word is always a scene onset
            #     # Check for changes in scene between consecutive words
            #     scene_changes = scene_values[1:] != scene_values[:-1]
            #     is_scene_onset[1:] = scene_changes
                
            #     self.label_indices = {
            #         1: np.where(is_scene_onset)[0], # positive indices: words at scene onsets
            #         0: np.arange(len(self.nonverbal_df)) # negative indices: nonverbal data
            #     }
        elif eval_name in ["scene_onset", "scene"]:
            # Detect scene changes: check if a scene switch occurs within each window
            # Scene labels are stored in scene_annotations.json in ROOT_DIR
            
            # Load scene_annotations.json file
            scene_annotations_path = os.path.join(ROOT_DIR, 'scene_annotations', 'scene_annotations.json')
            
            if not os.path.exists(scene_annotations_path):
                raise ValueError(f"Scene annotations file not found at: {scene_annotations_path}")
            
            with open(scene_annotations_path, 'r') as f:
                scene_annotations_data = json.load(f)
            
            # The file is a list with 1 item (a dict)
            if not isinstance(scene_annotations_data, list) or len(scene_annotations_data) == 0:
                raise ValueError(f"Scene annotations file has unexpected format: expected a list with 1 dict")
            
            annotations_dict = scene_annotations_data[0]
            
            # Find the key that matches this movie
            # Keys are like 'movie:annotations:v3:cars-2:jkdewit' or 'movie:annotations:v3:thor-ragnarok:stanford'
            # There may be multiple annotators, so prioritize 'all-annotations' if it exists
            movie_key = None
            all_annotations_key = None
            individual_keys = []

            for key in annotations_dict.keys():
                if f':{self.movie_name}:' in key or key.endswith(f':{self.movie_name}'):
                    if 'all-annotations' in key:
                        all_annotations_key = key
                    else:
                        individual_keys.append(key)

            # Prefer all-annotations version if it exists, otherwise use first individual annotator
            if all_annotations_key:
                movie_key = all_annotations_key
            elif len(individual_keys) > 0:
                movie_key = individual_keys[0]
                # Optionally warn if multiple annotators exist
                if len(individual_keys) > 1:
                    import warnings
                    warnings.warn(f"Multiple annotators found for '{self.movie_name}': {individual_keys}. Using: {movie_key}")

            if movie_key is None:
                raise ValueError(f"Could not find scene annotations for movie '{self.movie_name}' in scene_annotations.json")
            # Get the annotations for this movie
            movie_annotations = annotations_dict[movie_key]
            
            # Parse the annotations: keys are JSON strings like '{"startTime":2433,"endTime":2433,"label":"#"}'
            # Values are the startTime as strings
            # Create a mapping from time (seconds, rounded to int) to label
            scene_labels_dict = {}
            for json_key_str, start_time_str in movie_annotations.items():
                try:
                    # Parse the JSON key to get startTime and label
                    annotation = json.loads(json_key_str)
                    start_time = int(round(float(annotation.get('startTime', start_time_str))))
                    label = annotation.get('label', '')
                    
                    # Store the label for this time (use the most recent label if multiple entries for same second)
                    scene_labels_dict[start_time] = label
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    # Skip malformed entries
                    continue
            
            if len(scene_labels_dict) == 0:
                raise ValueError(f"No valid scene annotations found for movie '{self.movie_name}'")


            if eval_name == "scene":
                self.all_words_df["scene"] = self.all_words_df["start"].astype(int).map(scene_labels_dict)
                scene_labels = self.all_words_df["scene"].to_numpy()
                valid_mask = pd.notna(scene_labels) & (scene_labels != "") & (scene_labels != "#")
                valid_scene_labels = scene_labels[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                unique_scenes, counts = np.unique(valid_scene_labels, return_counts=True)
                
                # Select top K scenes
                if self.binary_tasks:
                    top_k = 2
                else:
                    top_k = 5  # or 10, 20, etc.
                
                top_k_indices = np.argsort(counts)[-top_k:][::-1]
                top_scenes = unique_scenes[top_k_indices]
                
                # Create label_indices
                top_scene_mask = np.isin(valid_scene_labels, top_scenes)
                filtered_indices = valid_indices[top_scene_mask]
                filtered_scenes = valid_scene_labels[top_scene_mask]
                
                self.label_indices = {}
                for label_id, scene in enumerate(top_scenes):
                    self.label_indices[label_id] = filtered_indices[np.where(filtered_scenes == scene)[0]]


            elif eval_name == "scene_onset":
                
                # Find scene boundaries (where scene changes between consecutive seconds)
                # Also consider "#" labels as explicit scene markers
                scene_times = sorted(scene_labels_dict.keys())
                scene_boundaries = []
                for i in range(1, len(scene_times)):
                    prev_time = scene_times[i-1]
                    curr_time = scene_times[i]
                    prev_label = scene_labels_dict[prev_time]
                    curr_label = scene_labels_dict[curr_time]
                    if prev_label != curr_label:
                        scene_boundaries.append(curr_time)
                scene_boundaries = sorted(list(set(scene_boundaries)))
                
                # Convert window parameters from samples to seconds
                window_start_seconds = self.start_neural_data_before_word_onset / SAMPLING_RATE
                window_end_seconds = self.end_neural_data_after_word_onset / SAMPLING_RATE
                
                # For each word, check if a scene boundary falls within its window
                has_scene_switch = []
                for idx, row in self.all_words_df.iterrows():
                    word_start_time = row['start']  # in seconds
                    window_start = word_start_time - window_start_seconds
                    window_end = word_start_time + window_end_seconds
                    
                    # Check if any scene boundary falls within this window
                    scene_switch_in_window = any(
                        window_start <= boundary <= window_end 
                        for boundary in scene_boundaries
                    )
                    has_scene_switch.append(scene_switch_in_window)
                
                has_scene_switch = np.array(has_scene_switch)
                
                # Positive samples: words whose windows contain scene switches
                # Negative samples: nonverbal windows (consistent with onset/speech)
                self.label_indices = {
                    1: np.where(has_scene_switch)[0],  # positive indices: words with scene switches in their windows
                    0: np.where(~has_scene_switch)[0]  # negative indices: nonverbal data
                }
        elif eval_name in ["face_num", "delta_face_num", "delta_face_num_abs", "delta_delta_face_num", "delta_delta_face_num_abs"]:
            face_nums = self.all_words_df["face_num"].to_numpy().astype(int)
            if eval_name == "delta_face_num":
                face_nums = self.all_words_df["delta_face_num"].to_numpy().astype(int)
            elif eval_name == "delta_face_num_abs":
                face_nums = self.all_words_df["delta_face_num_abs"].to_numpy().astype(int)
            elif eval_name == "delta_delta_face_num":
                face_nums = self.all_words_df["delta_delta_face_num"].to_numpy().astype(int)
            elif eval_name == "delta_delta_face_num_abs":
                face_nums = self.all_words_df["delta_delta_face_num_abs"].to_numpy().astype(int)
            if self.binary_tasks:
                self.label_indices = {
                    1: np.where(face_nums > 0)[0],
                    0: np.where(face_nums == 0)[0]
                }
            else:
                self.label_indices = {
                    2: np.where(face_nums > 1)[0],
                    1: np.where(face_nums == 1)[0],
                    0: np.where(face_nums == 0)[0]
                }
        elif eval_name == "word_index":
            word_indices = self.all_words_df["idx_in_sentence"].to_numpy().astype(int)
            if self.binary_tasks:
                self.label_indices = {
                    1: np.where(word_indices == 0)[0],
                    0: np.where(word_indices == 1)[0]
                }
            else:
                self.label_indices = {
                    2: np.where(word_indices >= 2)[0],
                    1: np.where(word_indices == 1)[0],
                    0: np.where(word_indices == 0)[0]
                }
        elif eval_name == "word_head_pos":
            head_pos = self.all_words_df[self.eval_name_remapped].to_numpy().astype(int)
            self.label_indices = {
                1: np.where(head_pos == 0)[0],
                0: np.where(head_pos == 1)[0]
            }
        elif eval_name == "word_part_speech":
            pos = self.all_words_df[self.eval_name_remapped].to_numpy()  
            if self.binary_tasks: 
                self.label_indices = {
                    1: np.where(pos == "VERB")[0],
                    0: np.where(pos == "NOUN")[0]
                }
            else:
                self.label_indices = {
                    5: np.where(pos == "ADV")[0],
                    4: np.where(pos == "ADJ")[0],
                    3: np.where(pos == "DET")[0],
                    2: np.where(pos == "PRON")[0],
                    1: np.where(pos == "VERB")[0],
                    0: np.where(pos == "NOUN")[0]
                }
        elif eval_name == "word_gap":
            word_gap_distribution = []
            for i in range(1, len(self.all_words_df)):
                if self.all_words_df.iloc[i]['sentence'] != self.all_words_df.iloc[i-1]['sentence']: continue
                gap = self.all_words_df.iloc[i]['start'] - self.all_words_df.iloc[i-1]['end']
                word_gap_distribution.append(gap)
            word_gap_distribution = np.array(word_gap_distribution)

            positive_indices = []
            negative_indices = []
            middle_indices = []
            for i in range(1, len(self.all_words_df)):
                if self.all_words_df.iloc[i]['sentence'] != self.all_words_df.iloc[i-1]['sentence']: continue
                gap = self.all_words_df.iloc[i]['start'] - self.all_words_df.iloc[i-1]['end']
                gap_percentile = np.mean(word_gap_distribution < gap)
                if gap_percentile >= 0.75:
                    positive_indices.append(i)
                elif (gap_percentile >= 0.375) and (gap_percentile < 0.625):
                    middle_indices.append(i)
                elif gap_percentile < 0.25:
                    negative_indices.append(i)
            if self.binary_tasks:
                self.label_indices = {
                    1: positive_indices,
                    0: negative_indices
                }
            else:
                self.label_indices = {
                    2: positive_indices,
                    1: middle_indices,
                    0: negative_indices
                }
        elif eval_name == "speaker":
            # Get speaker values and filter out invalid ones
            speakers = self.all_words_df["speaker"].to_numpy()
            
            # Filter out NaN, empty strings, and special values
            valid_mask = pd.notna(speakers) & (speakers != "") & (speakers != "* multiple speakers")
            valid_speakers = speakers[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            # Count speaker frequencies
            unique_speakers, counts = np.unique(valid_speakers, return_counts=True)
            
            # Select top K speakers based on binary_tasks flag
            if self.binary_tasks:
                top_k = 2
            else:
                top_k = 5  # or whatever K you want
            
            # Get top K most frequent speakers
            top_k_indices = np.argsort(counts)[-top_k:][::-1]  # Sort descending
            top_speakers = unique_speakers[top_k_indices]
            
            # Filter to only include words from top K speakers
            top_speaker_mask = np.isin(valid_speakers, top_speakers)
            filtered_indices = valid_indices[top_speaker_mask]
            filtered_speakers = valid_speakers[top_speaker_mask]
            
            # Create label_indices dictionary: map label_id -> array of word indices
            self.label_indices = {}
            for label_id, speaker in enumerate(top_speakers):
                self.label_indices[label_id] = filtered_indices[np.where(filtered_speakers == speaker)[0]]

        elif eval_name == "speaker_gender":
            self.all_words_df["speaker_gender"] = self.all_words_df["speaker"].map(speaker_gender).fillna("unknown")
            speaker_genders = self.all_words_df["speaker_gender"].to_numpy()
            self.label_indices = {
                1: np.where(speaker_genders == "male")[0],
                0: np.where(speaker_genders == "female")[0]
            }
        else:
            raise ValueError(f"Invalid eval_name: {eval_name}")

        self.n_classes = len(self.label_indices)
        n_samples_each = min([len(self.label_indices[label]) for label in self.label_indices])
        if self.lite: 
            n_samples_each = min(n_samples_each, NEUROPROBE_LITE_MAX_SAMPLES//self.n_classes)
        elif self.nano:
            n_samples_each = min(n_samples_each, NEUROPROBE_NANO_MAX_SAMPLES//self.n_classes)
        for label in list(self.label_indices.keys()):
            self.label_indices[label] = np.sort(self.rng.choice(self.label_indices[label], size=n_samples_each, replace=False))
            if self.max_samples is not None: # if max_samples is set, we need to truncate the indices to the max_samples
                self.label_indices[label] = self.label_indices[label][:self.max_samples//self.n_classes]
        self.n_samples = sum([len(self.label_indices[label]) for label in self.label_indices])

        self.cache_window_from = None
        self.cache_window_to = None
        if not self.always_cache_full_subject:
            n_try_indices = self.n_classes # try some first and last samples to get a good estimate of the edges of the needed data in the dataset
            window_indices = []
            for i in list(range(n_try_indices))+list(range(self.n_samples-n_try_indices, self.n_samples)):
                # (window_from, window_to), _ = self.__getitem__(i, force_output_indices=True)
                if self.output_dict:
                    window_from, window_to = self.__getitem__(i, force_output_indices=True)['data']
                else:
                    (window_from, window_to), _ = self.__getitem__(i, force_output_indices=True)
                window_indices.append(window_from)
                window_indices.append(window_to)
            self.cache_window_from = np.min(window_indices)
            self.cache_window_to = np.max(window_indices)

        

    def _get_neural_data(self, window_from, window_to, force_output_indices=False):
        self.subject.load_neural_data(self.trial_id, cache_window_from=self.cache_window_from, cache_window_to=self.cache_window_to)
        if not self.output_indices and not force_output_indices:
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=window_from, window_to=window_to)
            if self.lite or self.nano:
                input = input[self.electrode_indices_subset]
            return input.to(dtype=self.dtype)
        else:
            return window_from, window_to # just return the window indices

    def _positive_negative_getitem__(self, idx, force_output_indices=False):
        # even indices are positive samples, odd indices are negative samples
        current_label = (idx+1) % self.n_classes
        word_index = self.label_indices[current_label][idx//self.n_classes]
        # if self.eval_name == "scene_onset":
        #     row = self.scene_onset_windows_df.iloc[word_index]
        if self.eval_name in ["onset", "speech"] and (current_label == 0): # for onset and speech, we need to get the nonverbal data
            row = self.nonverbal_df.iloc[word_index]
        else:
            row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
        est_end_idx = est_idx + int(self.start_neural_data_before_word_onset) + int(self.end_neural_data_after_word_onset)
        input = self._get_neural_data(est_idx, est_end_idx, force_output_indices=force_output_indices)
        return input, current_label
        
        
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx, force_output_indices=False):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.n_samples}")
        input, label = self._positive_negative_getitem__(idx, force_output_indices=force_output_indices)
        
        if self.output_dict:
            return {
                "data": input, 
                "label": label, 
                "electrode_labels": self.electrode_labels,
                "electrode_coordinates": self.electrode_coordinates,
                "metadata": {
                    "dataset_identifier": "braintreebank",
                    "subject_id": self.subject.subject_id,
                    "trial_id": self.trial_id,
                    "sampling_rate": 2048,
                }
            }
        else:
            return input, label
