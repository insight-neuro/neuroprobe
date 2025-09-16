# Part of the code is adapted from https://braintreebank.dev/, file "quickstart.ipynb"
import os
import json
import pandas as pd
import numpy as np

# Add the parent directory to the path so that we can import the neuroprobe package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroprobe.config import *

def obtain_aligned_words_df(sub_id, trial_id, verbose=True, save_to_dir=None):
    # Data frames column IDs
    start_col, end_col = 'start', 'end'
    trig_time_col, trig_idx_col, est_idx_col, est_end_idx_col = 'movie_time', 'index', 'est_idx', 'est_end_idx'

    # Path to trigger times csv file
    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{sub_id}_trial{trial_id:03}_timings.csv')
    # Path format to trial metadata json file
    metadata_file = os.path.join(ROOT_DIR, f'subject_metadata/sub_{sub_id}_trial{trial_id:03}_metadata.json')
    with open(metadata_file, 'r') as f:
        meta_dict = json.load(f)
        title = meta_dict['title']
        movie_id = meta_dict['filename']
    # # Path to transcript csv file
    transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{movie_id}/features.csv')
    # Path format to electrode labels file -- mapping each ID to an subject specific label
    electrode_labels_file = os.path.join(ROOT_DIR, f'electrode_labels/sub_{sub_id}/electrode_labels.json')

    if verbose: print(f"Working with movie {title} ({movie_id})")

    if verbose: print(f"Computing words dataframe for subject {sub_id} trial {trial_id}")
    trigs_df = pd.read_csv(trigger_times_file)
    words_df = pd.read_csv(transcript_file_format.format(movie_id)).set_index('Unnamed: 0')
    words_df = words_df.drop(['word_diff', 'onset_diff'], axis=1) # remove those columns because they are unnecessary and cause excessive filtering with NaN values

    # Fix the 'sentence' column to never split words like "gonna" into "gon na". NOTE: actually no need to do it here, the full_word column fixes it later.
    # words_df['sentence'] = words_df['sentence'].str.replace(' na ', 'na ').replace(' na.', 'na.').replace(' na?', 'na?').replace(' na!', 'na!')
    # words_df['sentence'] = words_df['sentence'].str.replace(' ta ', 'ta ').replace(' ta.', 'ta.').replace(' ta?', 'ta?').replace(' ta!', 'ta!')

    # Add context (this is to provide the sentence just before the word) 
    # and full_word column (words before and after). This is to handle "split-word" parts like I'm, I've, I'd, gonna, can't etc.
    words_df['context'] = ''
    words_df['char_in_sentence'] = 0
    words_df['full_word'] = ''
    words_df['is_word_suffix'] = False
    for i in range(len(words_df)):
        sentence = words_df.loc[i, 'sentence']
        text = words_df.loc[i, 'text']
        if pd.isna(text): continue

        # Find character position in sentence
        if i > 0 and words_df.loc[i, 'sentence'] == words_df.loc[i-1, 'sentence']:
            # Same sentence as previous word, search after previous word's position
            prev_char_pos = int(words_df.loc[i-1, 'char_in_sentence'])
            char_in_sentence = sentence[prev_char_pos:].find(text) + prev_char_pos
        else:
            # First word in sentence or different sentence
            char_in_sentence = sentence.find(text)
        char_in_sentence = int(char_in_sentence)
        
        words_df.loc[i, 'context'] = sentence[:char_in_sentence].strip()
        words_df.loc[i, 'full_word'] = text
        words_df.loc[i, 'char_in_sentence'] = int(char_in_sentence) + len(text)
        # if the next word starts with the current word, then it is part of the current word -- parts like I'm, I've, I'd, gonna, can't etc.
        if (i<len(words_df)-1):
            next_word_text = words_df.loc[i+1, 'text']
            current_word_start_time = words_df.loc[i, 'start']
            next_word_start_time = words_df.loc[i+1, 'start']
            current_word_end_time = words_df.loc[i, 'end']
            next_word_end_time = words_df.loc[i+1, 'end']
            if (type(next_word_text) == str and 
                len(sentence) > char_in_sentence + len(text)):
                if sentence[char_in_sentence + len(text):].startswith(next_word_text):
                    words_df.loc[i, 'full_word'] = text + next_word_text
                    words_df.loc[i+1, 'is_word_suffix'] = True
                    if verbose: print(f"On index {i+1}, word {next_word_text} is a suffix of {text}")
                if next_word_start_time==current_word_start_time or next_word_end_time==current_word_end_time:
                    # No need to duplicate the word here.
                    # words_df.loc[i, 'full_word'] = text + next_word_text

                    words_df.loc[i+1, 'is_word_suffix'] = True
                    if verbose: print(f"On index {i+1}, word {next_word_text} is a correction of {text}")

    # Store original index before any transformations
    words_df['original_index'] = words_df.index.copy()
    # Remove rows where is_word_suffix is True
    words_df = words_df[~words_df['is_word_suffix']].copy()
    # Remove rows where there are NaN values
    words_df = words_df.dropna()
    
    # Assert no duplicate values in any of the movie time columns
    for col in ['start', 'end']:
        duplicates = words_df[col].duplicated()
        if duplicates.any():
            duplicate_indices = words_df[duplicates]['original_index'].tolist()
            print(f"WARNING: Found duplicate values in column {col} at indices: {duplicate_indices}. Removing these rows.")
            words_df = words_df[~duplicates].copy()
        
    words_df = words_df.reset_index(drop=True)

    # Vectorized sample index estimation
    def add_estimated_sample_index_vectorized(w_df, t_df):
        last_t = t_df[trig_time_col].iloc[-1]
        last_t_idx = t_df[trig_idx_col].idxmax()
        w_df = w_df[w_df[start_col] < last_t].copy()

        # Vectorized nearest trigger finding
        start_indices = np.searchsorted(t_df[trig_time_col].values, w_df[start_col].values)
        end_indices = np.searchsorted(t_df[trig_time_col].values, w_df[end_col].values)
        end_indices = np.minimum(end_indices, last_t_idx) # handle the edge case where movie cuts off right at the word
        start_indices = np.maximum(start_indices, 0) # handle the edge case where movie starts right at the word
        
        # Vectorized sample index calculation
        w_df[est_idx_col] = np.round(
            t_df.loc[start_indices, trig_idx_col].values + 
            (w_df[start_col].values - t_df.loc[start_indices, trig_time_col].values) * SAMPLING_RATE
        )
        w_df[est_end_idx_col] = np.round(
            t_df.loc[end_indices, trig_idx_col].values + 
            (w_df[end_col].values - t_df.loc[end_indices, trig_time_col].values) * SAMPLING_RATE
        )
        return w_df

    words_df = add_estimated_sample_index_vectorized(words_df, trigs_df)  # align all words to data samples
    words_df = words_df.dropna().reset_index(drop=True)  # no need to keep words with no start time

    # Remove words that would create invalid windows (too close to the start or end of the trial)
    total_samples = trigs_df.loc[len(trigs_df) - 1, trig_idx_col]
    valid_words = (words_df[est_idx_col] >= int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE)) & \
                 (words_df[est_end_idx_col] <= int(total_samples - END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE))
    words_df = words_df[valid_words].reset_index(drop=True)

    # Keep only the columns that were added in the code and the original index
    columns_to_keep = ['original_index', est_idx_col, est_end_idx_col, start_col, end_col, 'context', 'char_in_sentence', 'full_word']
    words_df = words_df[columns_to_keep]

    if verbose: print(f"Kept {len(words_df)} words after removing invalid windows")
    # Save the processed words dataframe
    if save_to_dir is not None:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        words_df.to_csv(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_words_df.csv', index=False)
    return words_df


def obtain_nonverbal_df(sub_id, trial_id, words_df, verbose=True, save_to_dir=None):
    # Data frames column IDs
    start_col, end_col, lbl_col = 'start', 'end', 'pos'
    trig_time_col, trig_idx_col, est_idx_col, est_end_idx_col = 'movie_time', 'index', 'est_idx', 'est_end_idx'
    word_time_col, word_text_col, is_onset_col, is_offset_col = 'word_time', 'text', 'is_onset', 'is_offset'

    window_length = int((START_NEURAL_DATA_BEFORE_WORD_ONSET + END_NEURAL_DATA_AFTER_WORD_ONSET) * SAMPLING_RATE)

    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{sub_id}_trial{trial_id:03}_timings.csv')
    trigs_df = pd.read_csv(trigger_times_file)
    total_samples = trigs_df.loc[len(trigs_df) - 1, trig_idx_col]

    nonverbal_windows = []
    # Iterate through consecutive word pairs
    for i in range(len(words_df)-1):
        current_word = words_df.iloc[i]
        next_word = words_df.iloc[i+1]
        
        # Calculate gap between current word offset and next word onset
        current_word_offset = int(current_word[est_end_idx_col] + int((NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME+START_NEURAL_DATA_BEFORE_WORD_ONSET) * SAMPLING_RATE))
        next_word_onset = int(next_word[est_idx_col] - int((NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME+END_NEURAL_DATA_AFTER_WORD_ONSET) * SAMPLING_RATE))
        gap_samples = next_word_onset - current_word_offset

        window_start_sample = current_word_offset
        window_end_sample = window_start_sample + window_length

        # If gap is large enough for a window
        while (gap_samples >= window_length) and (window_end_sample <= total_samples):
            # Add new row to nonverbal_df with window start/end times and sample indices
            nonverbal_windows.append({
                start_col: window_start_sample / SAMPLING_RATE,
                end_col: window_end_sample / SAMPLING_RATE, 
                est_idx_col: window_start_sample,
                est_end_idx_col: window_end_sample
            })
            window_start_sample += int(window_length * (1 - NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP))
            window_end_sample += int(window_length * (1 - NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP))
            gap_samples -= int(window_length * (1 - NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP))
    nonverbal_df = pd.DataFrame(nonverbal_windows)        
    print(f"Kept {len(nonverbal_df)} nonverbal windows")
    if save_to_dir is not None:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        nonverbal_df.to_csv(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_nonverbal_df.csv', index=False)
    return nonverbal_df


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, help='Subject ID (optional)')
    parser.add_argument('--trial', type=int, help='Trial ID (optional)') 
    args = parser.parse_args()

    all_subject_trials = NEUROPROBE_FULL_SUBJECT_TRIALS

    # If subject and trial specified, only process that pair
    if args.subject is not None and args.trial is not None:
        subject_trials = [(args.subject, args.trial)]
    else:
        subject_trials = all_subject_trials

    for sub_id, trial_id in subject_trials:
        words_df = obtain_aligned_words_df(sub_id, trial_id, save_to_dir=SAVE_SUBJECT_TRIAL_DF_DIR)
        nonverbal_df = obtain_nonverbal_df(sub_id, trial_id, words_df, save_to_dir=SAVE_SUBJECT_TRIAL_DF_DIR)