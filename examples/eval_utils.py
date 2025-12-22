import neuroprobe.config as neuroprobe_config

from sklearn.metrics import roc_auc_score
import torch, numpy as np
import time, psutil
import math

verbose = True # print logs

# -------------------- DAE compatibility toggle --------------------
# NeuroProbe's STFT preprocessing (`preprocess_stft`) returns tensors shaped:
#   (batch, electrodes, timebins, freqs)
# But the 2D conv paths in the autoencoders below assume:
#   (batch, electrodes, freqs, timebins)
#
# Set this to False (or comment out the transpose blocks marked "DAE_STFT_AXIS_FIX")
# to revert to the historical behavior.
DAE_FIX_STFT_AXIS_ORDER = False

# -------------------- GNN compatibility toggle --------------------
# NeuroProbe's STFT preprocessing (`preprocess_stft`) returns tensors shaped:
#   (batch, electrodes, timebins, freqs)
# But `GNNClassifier`'s STFT path assumes:
#   (batch, electrodes, freqs, timebins)
#
# Set this to False (or comment out the blocks marked "GNN_STFT_AXIS_FIX")
# to revert to the historical behavior.
GNN_FIX_STFT_AXIS_ORDER = False

############## LOGGING ###############

def model_name_from_classifier_type(classifier_type):
    if classifier_type == 'linear':
        return "Logistic Regression"
    elif classifier_type == 'cnn':
        return "CNN"
    elif classifier_type == 'transformer':
        return "Transformer"
    elif classifier_type == 'mlp':
        return "MLP"
    elif classifier_type == 'hybrid':
        return "CNN-RNN Hybrid"
    elif classifier_type == 'gnn':
        return "Graph Neural Network"
    elif classifier_type == 'dae':
        return "Denoising Autoencoder"
    elif classifier_type == 'vae':
        return "Variational Autoencoder"
    elif classifier_type == 'brainbert':
        return "BrainBERT"
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")

def log(message, priority=0, indent=0):
    max_log_priority = -1 if not verbose else 4
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}")


############## ELECTRODE SUBSET ###############

def subset_electrodes(subject, lite=False, nano=False):
    all_electrode_labels = subject.electrode_labels
    if lite:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
    elif nano:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
    subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
    return all_electrode_labels


############## DATA PREPROCESSING ###############

from scipy import signal
import numpy as np

def preprocess_stft(data, sampling_rate=2048, preprocess="stft_abs", 
                    preprocess_parameters={"stft": {"nperseg": 512, "poverlap": 0.75, "window": "hann", "max_frequency": 150, "min_frequency": 0}}):
    was_tensor = isinstance(data, torch.Tensor)
    x = torch.from_numpy(data) if not was_tensor else data

    if len(x.shape) == 2: # if it is only (n_electrodes, n_samples)
        x = x.unsqueeze(0)
    # data is of shape (batch_size, n_electrodes, n_samples)
    batch_size, n_electrodes, n_samples = x.shape

    # convert to float32 and reshape for STFT
    x = x.to(dtype=torch.float32)
    x = x.reshape(batch_size * n_electrodes, -1)

    # STFT parameters
    nperseg = preprocess_parameters["stft"]["nperseg"]
    poverlap = preprocess_parameters["stft"]["poverlap"]
    noverlap = int(nperseg * poverlap)
    hop_length = nperseg - noverlap

    if preprocess_parameters["stft"]["window"] == "hann":
        window = torch.hann_window(nperseg, device=x.device)
    elif preprocess_parameters["stft"]["window"] == "boxcar":
        window = torch.ones(nperseg, device=x.device)
    else:
        raise ValueError(f"Invalid window type: {preprocess_parameters['stft']['window']}")
    
    max_frequency = preprocess_parameters["stft"]["max_frequency"]
    min_frequency = preprocess_parameters["stft"]["min_frequency"]

    # Compute STFT
    x = torch.stft(x,
                    n_fft=nperseg, 
                    hop_length=hop_length,
                    win_length=nperseg,
                    window=window,
                    return_complex=True,
                    normalized=False,
                    center=True)
    # Get frequency bins
    freqs = torch.fft.rfftfreq(nperseg, d=1.0/sampling_rate) # 2048Hz sampling rate
    x = x[:, (freqs >= min_frequency) & (freqs <= max_frequency)]

    if preprocess == "stft_absangle":
        # Split complex values into magnitude and phase
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        # Stack magnitude and phase along a new axis
        x = torch.stack([magnitude, phase], dim=-2)
    elif preprocess == "stft_realimag":
        real = torch.real(x)
        imag = torch.imag(x)
        x = torch.stack([real, imag], dim=-2)
    elif preprocess == "stft_abs":   
        x = torch.abs(x)
    else:
        raise ValueError(f"Invalid preprocess type: {preprocess}")

    # Reshape back
    _, n_freqs, n_times = x.shape
    x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
    x = x.transpose(2, 3) # (batch_size, n_electrodes, n_timebins, n_freqs)
    
    # Z-score normalization
    # NOTE: skipping batch norm here because in the regression pipeline, StandardScaler is used anyway,
    # and we would like to avoid batch effects in case input items are processed one by one. TODO: find a better idea here
    # x = x - x.mean(dim=[0, 2], keepdim=True)
    # x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)

    return x.numpy() if not was_tensor else x

def downsample(data, fs=2048, downsample_rate=200):
    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        device = data.device
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    # Apply downsampling
    result = signal.resample_poly(data_np, up=downsample_rate, down=fs, axis=-1)
    
    # Convert back to tensor if input was a tensor
    if was_tensor:
        result = torch.from_numpy(result).to(device)
    
    return result
def remove_line_noise(data, fs=2048, line_freq=60):
    """Remove line noise (60 Hz and harmonics) from neural data.
    
    Args:
        data (numpy.ndarray or torch.Tensor): Input voltage data of shape (n_channels, n_samples) or (batch_size, n_channels, n_samples)
        fs (int): Sampling frequency in Hz
        line_freq (int): Fundamental line frequency in Hz (typically 60 Hz in the US)
        
    Returns:
        numpy.ndarray or torch.Tensor: Filtered data with the same shape as input (same type as input)
    """
    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        device = data.device
        filtered_data = data.cpu().numpy().copy()
    else:
        filtered_data = data.copy()
    
    # Define the width of the notch filter (5 Hz on each side)
    bandwidth = 5.0
    
    # Calculate the quality factor Q
    Q = line_freq / bandwidth
    
    # Apply notch filters for the fundamental frequency and harmonics
    # We'll filter up to the 5th harmonic (60, 120, 180, 240, 300 Hz)
    for harmonic in range(1, 6):
        harmonic_freq = line_freq * harmonic
        
        # Skip if the harmonic frequency is above the Nyquist frequency
        if harmonic_freq > fs/2:
            break
            
        # Create and apply a notch filter
        b, a = signal.iirnotch(harmonic_freq, Q, fs)
        
        # Apply the filter along the time dimension
        if filtered_data.ndim == 2:  # (n_channels, n_samples)
            filtered_data = signal.filtfilt(b, a, filtered_data, axis=1)
        elif filtered_data.ndim == 3:  # (batch_size, n_channels, n_samples)
            for i in range(filtered_data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, filtered_data[i], axis=1)
    
    # Convert back to tensor if input was a tensor
    if was_tensor:
        filtered_data = torch.from_numpy(filtered_data).to(device)
    
    return filtered_data

def laplacian_rereference_neural_data(electrode_data, electrode_labels, remove_non_laplacian=True):
    """
    Rereference the neural data using the laplacian method (subtract the mean of the neighbors, as determined by the electrode labels)
    inputs:
        electrode_data: torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
        electrode_labels: list of electrode labels
        remove_non_laplacian: boolean, if True, remove the non-laplacian electrodes from the data; if false, keep them without rereferencing
    outputs:
        rereferenced_data: torch tensor of shape (batch_size, n_electrodes_rereferenced, n_samples) or (n_electrodes_rereferenced, n_samples)
        rereferenced_labels: list of electrode labels of length n_electrodes_rereferenced (n_electrodes_rereferenced could be different from n_electrodes if remove_non_laplacian is True)
    """
    def get_all_laplacian_electrodes(electrode_labels):
        """
            Get all laplacian electrodes for a given subject. This function is originally from
            https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)
        """
        def stem_electrode_name(name):
            #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
            #names look like 'T1b2
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
            return ((x,y+1) in stems) or ((x,y-1) in stems)
        def get_neighbors(stem, stems):
            (x,y) = stem
            return [f'{x}{y}' for (x,y) in [(x,y+1), (x,y-1)] if (x, y) in stems]
        stems = [stem_electrode_name(e) for e in electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
        neighbors = {e: get_neighbors(stem_electrode_name(e), stems) for e in electrodes}
        return electrodes, neighbors

    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(electrode_data, torch.Tensor)

    batch_unsqueeze = False
    if len(electrode_data.shape) == 2:
        batch_unsqueeze = True
        if was_tensor:
            electrode_data = electrode_data.unsqueeze(0)
        else:
            electrode_data = electrode_data[np.newaxis, :, :]

    laplacian_electrodes, laplacian_neighbors = get_all_laplacian_electrodes(electrode_labels)
    laplacian_neighbor_indices = {laplacian_electrode_label: [electrode_labels.index(neighbor_label) for neighbor_label in neighbors] for laplacian_electrode_label, neighbors in laplacian_neighbors.items()}

    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    if was_tensor:
        rereferenced_data = torch.zeros((batch_size, rereferenced_n_electrodes, n_samples), dtype=electrode_data.dtype, device=electrode_data.device)
    else:
        rereferenced_data = np.zeros((batch_size, rereferenced_n_electrodes, n_samples), dtype=electrode_data.dtype)

    electrode_i = 0
    original_electrode_indices = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        if electrode_label in laplacian_electrodes:
            rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i] - electrode_data[:, laplacian_neighbor_indices[electrode_label]].mean(axis=1)
            original_electrode_indices.append(original_electrode_index)
            electrode_i += 1
        else:
            if remove_non_laplacian: 
                continue # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i]
                original_electrode_indices.append(original_electrode_index)
                electrode_i += 1
                
    if batch_unsqueeze:
        if was_tensor:
            rereferenced_data = rereferenced_data.squeeze(0)
        else:
            rereferenced_data = rereferenced_data.squeeze(0)

    return rereferenced_data, laplacian_electrodes if remove_non_laplacian else electrode_labels, original_electrode_indices

def preprocess_data(data, electrode_labels, preprocess, preprocess_parameters):
    for preprocess_option in preprocess.split('-'):
        if preprocess_option.lower() in ['stft_absangle', 'stft_realimag', 'stft_abs']:
            data = preprocess_stft(data, preprocess=preprocess_option, preprocess_parameters=preprocess_parameters)
        elif preprocess_option.lower() == 'remove_line_noise':
            data = remove_line_noise(data)
        elif preprocess_option.lower() == 'downsample_200':
            data = downsample(data, downsample_rate=200)
        elif preprocess_option.lower() == 'downsample_500':
            data = downsample(data, downsample_rate=500)
        elif preprocess_option.lower() == 'laplacian':
            data, electrode_labels, original_electrode_indices = laplacian_rereference_neural_data(data, electrode_labels, remove_non_laplacian=False)
    return data



############## CLASSIFICATION ###############


class TransformerClassifier:
    def __init__(self, random_state=42, max_iter=100, batch_size=64, learning_rate=0.001, val_size=0.2, tol=1e-4, patience=10,
                 d_model=64, nhead=8, dim_feedforward=256, dropout=0.1, num_layers=3):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers

    def _create_model(self, input_shape, n_classes):
        class Transformer(torch.nn.Module):
            def __init__(self, input_shape, n_classes, d_model=64, nhead=8, dim_feedforward=256, dropout=0.1, num_layers=3):
                super().__init__()
                self.d_model = d_model
                self.nhead = nhead
                self.dim_feedforward = dim_feedforward
                self.dropout = dropout
                self.num_layers = num_layers
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    self.input_proj = torch.nn.Linear(input_shape[0], self.d_model)  # Project channels to embedding dim
                    self.pos_encoder = PositionalEncoding(self.d_model, max_len=input_shape[1])
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                    self.fc = torch.nn.Linear(self.d_model, n_classes)
                else:  # 3D input (channels, freq, time)
                    self.input_proj = torch.nn.Linear(input_shape[0] * input_shape[1], self.d_model)  # Project channels*freq to embedding dim
                    self.pos_encoder = PositionalEncoding(self.d_model, max_len=input_shape[2])
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                    self.fc = torch.nn.Linear(self.d_model, n_classes)
                
            def forward(self, x):
                # Reshape input for transformer
                if len(x.shape) == 3:  # (batch, channels, time)
                    x = x.transpose(1, 2)  # (batch, time, channels)
                    x = self.input_proj(x)  # (batch, time, 64)
                else:  # (batch, channels, freq, time)
                    batch_size, channels, freq, time = x.shape
                    x = x.transpose(1, 3)  # (batch, time, channels, freq)
                    x = x.reshape(batch_size, time, channels * freq)
                    x = self.input_proj(x)  # (batch, time, 64)
                
                # Add positional encoding
                x = self.pos_encoder(x)
                
                # Apply transformer
                x = self.transformer_encoder(x)
                
                # Global average pooling over time dimension
                x = x.mean(dim=1)
                
                # Final classification layer
                x = self.fc(x)
                return x
        
        class PositionalEncoding(torch.nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        
        return Transformer(input_shape, n_classes, d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, num_layers=self.num_layers)
    
    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create train/val split - take last portion for validation
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                log(f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                        break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class CNNClassifier:
    def __init__(self, random_state=42, max_iter=100, batch_size=128, learning_rate=0.0001, val_size=0.2, tol=1e-4, patience=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        
    def _create_model(self, input_shape, n_classes):
        class CNN(torch.nn.Module):
            def __init__(self, input_shape, n_classes):
                super().__init__()
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    self.conv1 = torch.nn.Conv1d(input_shape[0], 32, kernel_size=3, padding=1)
                    self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool1d(2)
                    self.dropout = torch.nn.Dropout(0.5)
                    
                    # Calculate the size after convolutions and pooling
                    conv_output_size = input_shape[1] // 8 * 128
                    
                    self.fc1 = torch.nn.Linear(conv_output_size, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)
                    
                else:  # 3D input (channels, freq, time)
                    self.conv1 = torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
                    self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool2d(2)
                    self.dropout = torch.nn.Dropout(0.5)
                    
                    # Calculate the size after convolutions and pooling
                    conv_output_size = (input_shape[1] // 8) * (input_shape[2] // 8) * 128
                    
                    self.fc1 = torch.nn.Linear(conv_output_size, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)
                
                self.relu = torch.nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return CNN(input_shape, n_classes)
    
    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create train/val split - take last portion for validation
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                log(f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                        break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

class MLPClassifier:
    def __init__(self, random_state=42, max_iter=100, batch_size=200, learning_rate=0.00001, hidden_dims=[1024, 1024],
                 tol=1e-8, patience=100):
        """
        MLP Classifier with configurable hidden layers.
        
        Args:
            hidden_dims: list of integers specifying hidden layer dimensions.
                        If None or empty list, creates a linear model (no hidden layers).
                        E.g., [128, 64] creates two hidden layers with 128 and 64 units.
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tol = tol
        self.patience = patience
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        
    def _create_model(self, input_size, n_classes):
        class MLP(torch.nn.Module):
            def __init__(self, input_size, n_classes, hidden_dims):
                super().__init__()
                layers = []
                
                if len(hidden_dims) == 0:
                    # Linear model (logistic regression)
                    layers.append(torch.nn.Linear(input_size, n_classes))
                else:
                    # MLP with hidden layers
                    prev_dim = input_size
                    for hidden_dim in hidden_dims:
                        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                        layers.append(torch.nn.ReLU())
                        layers.append(torch.nn.Dropout(0.2))
                        prev_dim = hidden_dim
                    
                    # Output layer
                    layers.append(torch.nn.Linear(prev_dim, n_classes))
                
                self.network = torch.nn.Sequential(*layers)
                
            def forward(self, x):
                # Flatten all dimensions except batch
                x = x.view(x.size(0), -1)
                return self.network(x)
        
        return MLP(input_size, n_classes, self.hidden_dims)
    
    def fit(self, X, y, X_val=None, y_val=None):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        model_type = "Linear" if len(self.hidden_dims) == 0 else f"MLP{self.hidden_dims}"
        log(f"Training {model_type} on full dataset with {len(X)} samples", priority=3, indent=2)

        # Create model - flatten input to get total size
        input_size = np.prod(X.shape[1:])
        self.model = self._create_model(input_size, n_classes)
        self.model = self.model.to(self.device)

        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_model_state = None
        patience_counter = 0
        best_val_auroc = 0.0

        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                batch_y = y[i:i+self.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss = train_loss / train_total
            train_probs = self.predict_proba(X)
            train_auroc = roc_auc_score(y, train_probs[:, 1], multi_class='ovr', average='macro')
            
            # Calculate val AUROC if val data is provided
            if X_val is not None and y_val is not None:
                all_val_probs = []
                with torch.no_grad():
                    for i in range(0, len(X_val), self.batch_size):
                        batch_X = X_val[i:i+self.batch_size].to(self.device)
                        outputs = self.model(batch_X)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_val_probs.append(probs.cpu().numpy())
                val_probs = np.concatenate(all_val_probs, axis=0)
                val_auroc = roc_auc_score(y_val, val_probs[:, 1], multi_class='ovr', average='macro')

            log(f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.8f}, Train AUROC: {train_auroc:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)

            # Early stopping based on validation AUROC
            if val_auroc > best_val_auroc + self.tol:
                best_val_auroc = val_auroc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class HybridCNNRNNClassifier:
    """
    Hybrid CNN-RNN classifier that combines:
    - CNN layers for spatial feature extraction across electrodes
    - RNN (LSTM) layers for temporal dynamics
    """
    def __init__(self, random_state=42, max_iter=100, batch_size=64, learning_rate=0.001, val_size=0.2, tol=1e-4, patience=10,
                 cnn_channels=[32, 64, 128], lstm_hidden=128, lstm_layers=2, dropout=0.3):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
    def _create_model(self, input_shape, n_classes):
        class HybridCNNRNN(torch.nn.Module):
            def __init__(self, input_shape, n_classes, cnn_channels=[32, 64, 128], lstm_hidden=128, lstm_layers=2, dropout=0.3):
                super().__init__()
                self.cnn_channels = cnn_channels
                self.lstm_hidden = lstm_hidden
                self.lstm_layers = lstm_layers
                self.dropout = dropout
                
                if len(input_shape) == 2:
                    # 2D input: (channels, time)
                    n_channels, n_time = input_shape
                    
                    # CNN layers for spatial feature extraction
                    self.conv_layers = torch.nn.ModuleList()
                    in_channels = n_channels
                    for out_channels in cnn_channels:
                        self.conv_layers.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
                        in_channels = out_channels
                    
                    # Calculate actual output size after CNN+pooling by doing a dummy forward pass
                    # We'll set this after the first forward pass
                    self.lstm_input_size = None
                    self.lstm = None  # Will be created after first forward pass
                    
                    # LSTM output is (batch, n_time, lstm_hidden * 2) for bidirectional
                    self.fc1 = torch.nn.Linear(lstm_hidden * 2, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)
                    
                else:  # 3D input: (channels, freq, time)
                    n_channels, n_freq, n_time = input_shape
                    
                    # CNN layers for spatial-spectral feature extraction
                    self.conv_layers = torch.nn.ModuleList()
                    in_channels = n_channels
                    for out_channels in cnn_channels:
                        self.conv_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                        in_channels = out_channels
                    
                    # Calculate actual output size after CNN+pooling by doing a dummy forward pass
                    # We'll set this after the first forward pass
                    self.lstm_input_size = None
                    self.lstm = None  # Will be created after first forward pass
                    
                    self.fc1 = torch.nn.Linear(lstm_hidden * 2, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)
                
                self.relu = torch.nn.ReLU()
                self.dropout_layer = torch.nn.Dropout(dropout)
                self.pool1d = torch.nn.MaxPool1d(2)
                self.pool2d = torch.nn.MaxPool2d(2)
                
            def _ensure_lstm(self, lstm_input_size):
                """Create LSTM if it doesn't exist or if input size changed"""
                if self.lstm is None or self.lstm_input_size != lstm_input_size:
                    self.lstm_input_size = lstm_input_size
                    self.lstm = torch.nn.LSTM(
                        input_size=lstm_input_size,
                        hidden_size=self.lstm_hidden,
                        num_layers=self.lstm_layers,
                        batch_first=True,
                        dropout=self.dropout if self.lstm_layers > 1 else 0,
                        bidirectional=True
                    )
                    # Move to same device as other modules
                    if next(self.conv_layers.parameters()).is_cuda:
                        self.lstm = self.lstm.cuda()
                
            def forward(self, x):
                # Apply CNN layers
                for i, conv in enumerate(self.conv_layers):
                    x = self.relu(conv(x))
                    if len(x.shape) == 3:  # 1D conv: (batch, channels, time)
                        # Use adaptive pooling to ensure we don't reduce to 0
                        if x.shape[2] > 2:
                            x = self.pool1d(x)
                        else:
                            # If too small, use adaptive pooling to get at least size 1
                            x = torch.nn.functional.adaptive_avg_pool1d(x, 1)
                    else:  # 2D conv: (batch, channels, freq, time)
                        # Check dimensions before pooling
                        if x.shape[2] > 2 and x.shape[3] > 2:
                            x = self.pool2d(x)
                        elif x.shape[2] == 1 and x.shape[3] > 2:
                            # Only time dimension can be pooled
                            x = torch.nn.functional.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
                        elif x.shape[2] > 2 and x.shape[3] == 1:
                            # Only freq dimension can be pooled
                            x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))
                        else:
                            # Both dimensions are small, use adaptive pooling
                            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, max(1, x.shape[3])))
                
                # Reshape for LSTM
                if len(x.shape) == 3:  # (batch, channels, time)
                    # Transpose to (batch, time, channels)
                    x = x.transpose(1, 2)
                    lstm_input_size = x.shape[2]  # channels dimension
                else:  # (batch, channels, freq, time)
                    # Reshape to (batch, time, channels * freq)
                    batch_size, channels, freq, time = x.shape
                    x = x.transpose(1, 3)  # (batch, time, channels, freq)
                    x = x.reshape(batch_size, time, channels * freq)
                    lstm_input_size = x.shape[2]  # channels * freq
                
                # Create LSTM with correct input size
                self._ensure_lstm(lstm_input_size)
                
                # Apply LSTM
                lstm_out, (h_n, c_n) = self.lstm(x)
                # Use the last output from LSTM
                x = lstm_out[:, -1, :]  # (batch, lstm_hidden * 2)
                
                # Final classification layers
                x = self.dropout_layer(x)
                x = self.relu(self.fc1(x))
                x = self.dropout_layer(x)
                x = self.fc2(x)
                
                return x
        
        return HybridCNNRNN(input_shape, n_classes, cnn_channels=self.cnn_channels, 
                           lstm_hidden=self.lstm_hidden, lstm_layers=self.lstm_layers, 
                           dropout=self.dropout)
    
    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create train/val split - take last portion for validation
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                log(f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                        break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class GNNClassifier:
    """
    Graph Neural Network classifier that models spatial relationships between electrodes.
    Uses Graph Convolutional Network (GCN) layers to process multi-electrode neural signals.
    """
    def __init__(self, random_state=42, max_iter=100, batch_size=64, learning_rate=0.001, val_size=0.2, tol=1e-4, patience=10,
                 gcn_hidden=[64, 128], dropout=0.3, k_neighbors=10, distance_threshold=None):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        self.gcn_hidden = gcn_hidden if isinstance(gcn_hidden, list) else [gcn_hidden]
        self.dropout = dropout
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.adjacency_matrix = None
        self.electrode_coordinates = None
        
    def _build_adjacency_matrix(self, coordinates, k_neighbors=None, distance_threshold=None):
        """
        Build adjacency matrix from electrode coordinates.
        
        Args:
            coordinates: (n_electrodes, 3) tensor of electrode coordinates
            k_neighbors: Number of nearest neighbors to connect (if None, use self.k_neighbors)
            distance_threshold: Maximum distance to connect electrodes (if None, use k_neighbors)
        
        Returns:
            adjacency_matrix: (n_electrodes, n_electrodes) sparse adjacency matrix
        """
        if k_neighbors is None:
            k_neighbors = self.k_neighbors
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
            
        n_electrodes = coordinates.shape[0]
        coordinates = coordinates.to(self.device)
        
        # Compute pairwise distances
        # coordinates: (n_electrodes, 3)
        # Expand to (n_electrodes, 1, 3) and (1, n_electrodes, 3) for broadcasting
        coords_expanded_1 = coordinates.unsqueeze(1)  # (n_electrodes, 1, 3)
        coords_expanded_2 = coordinates.unsqueeze(0)  # (1, n_electrodes, 3)
        distances = torch.sqrt(torch.sum((coords_expanded_1 - coords_expanded_2) ** 2, dim=2))  # (n_electrodes, n_electrodes)
        
        # Build adjacency matrix
        if distance_threshold is not None:
            # Connect electrodes within distance threshold
            adjacency = (distances <= distance_threshold).float()
        else:
            # Connect k nearest neighbors (including self)
            _, indices = torch.topk(distances, k=min(k_neighbors + 1, n_electrodes), dim=1, largest=False)
            adjacency = torch.zeros_like(distances)
            for i in range(n_electrodes):
                adjacency[i, indices[i]] = 1.0
        
        # Make symmetric (undirected graph)
        adjacency = (adjacency + adjacency.T) / 2
        adjacency = (adjacency > 0).float()
        
        # Add self-connections
        adjacency += torch.eye(n_electrodes, device=self.device)
        
        # Normalize adjacency matrix (symmetric normalization)
        degree = torch.sum(adjacency, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        normalized_adjacency = degree_inv_sqrt @ adjacency @ degree_inv_sqrt
        
        return normalized_adjacency
    
    def _create_model(self, input_shape, n_classes, adjacency_matrix):
        class GraphConvolution(torch.nn.Module):
            """Simple Graph Convolutional Layer"""
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
                self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
                self.reset_parameters()
            
            def reset_parameters(self):
                torch.nn.init.xavier_uniform_(self.weight)
                torch.nn.init.zeros_(self.bias)
            
            def forward(self, x, adjacency):
                # x: (batch, n_nodes, in_features)
                # adjacency: (n_nodes, n_nodes)
                # Output: (batch, n_nodes, out_features)
                support = torch.matmul(x, self.weight)  # (batch, n_nodes, out_features)
                output = torch.matmul(adjacency, support)  # (batch, n_nodes, out_features)
                return output + self.bias
        
        class GCN(torch.nn.Module):
            def __init__(self, input_shape, n_classes, adjacency_matrix, gcn_hidden=[64, 128], dropout=0.3):
                super().__init__()
                self.adjacency_matrix = adjacency_matrix
                self.dropout = dropout
                
                # Determine input feature size
                if len(input_shape) == 2:
                    # 2D input: (channels, time)
                    n_nodes, n_time = input_shape
                    input_features = 1  # Each node has 1 feature per time step
                    self.use_lstm = True  # Use LSTM for temporal modeling
                else:
                    # 3D input: (channels, freq, time)
                    n_nodes, n_freq, n_time = input_shape
                    input_features = n_freq  # Each node has n_freq features per time step
                    self.use_lstm = True
                    self.n_freq = n_freq
                
                # Build GCN layers for spatial feature extraction
                self.gcn_layers = torch.nn.ModuleList()
                layer_sizes = [input_features] + gcn_hidden
                
                for i in range(len(layer_sizes) - 1):
                    self.gcn_layers.append(GraphConvolution(layer_sizes[i], layer_sizes[i+1]))
                
                # LSTM for temporal modeling after GCN
                self.temporal_lstm = torch.nn.LSTM(gcn_hidden[-1], gcn_hidden[-1], 
                                                    num_layers=1, batch_first=True, bidirectional=True)
                
                # Final classification layers
                final_dim = gcn_hidden[-1] * 2  # *2 for bidirectional LSTM
                self.fc1 = torch.nn.Linear(final_dim, 256)
                self.fc2 = torch.nn.Linear(256, n_classes)
                self.dropout_layer = torch.nn.Dropout(dropout)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                # x: (batch, n_nodes, ...)
                batch_size = x.shape[0]
                
                if len(x.shape) == 3:
                    # 2D input: (batch, n_nodes, n_time)
                    batch_size, n_nodes, n_time = x.shape
                    # Process each time step through GCN
                    time_features = []
                    for t in range(n_time):
                        x_t = x[:, :, t].unsqueeze(-1)  # (batch, n_nodes, 1)
                        # Apply GCN layers for spatial feature extraction
                        for gcn_layer in self.gcn_layers:
                            x_t = gcn_layer(x_t, self.adjacency_matrix)
                            x_t = self.relu(x_t)
                            x_t = self.dropout_layer(x_t)
                        # Pool over nodes: (batch, n_nodes, hidden) -> (batch, hidden)
                        x_t = x_t.mean(dim=1)  # Global average pooling over electrodes
                        time_features.append(x_t)
                    
                    # Stack: (batch, n_time, hidden)
                    x = torch.stack(time_features, dim=1)
                    
                else:
                    # 3D input: (batch, n_nodes, n_freq, n_time)
                    batch_size, n_nodes, n_freq, n_time = x.shape
                    # Process each time step through GCN
                    time_features = []
                    for t in range(n_time):
                        x_t = x[:, :, :, t]  # (batch, n_nodes, n_freq)
                        # Apply GCN layers for spatial feature extraction
                        for gcn_layer in self.gcn_layers:
                            x_t = gcn_layer(x_t, self.adjacency_matrix)
                            x_t = self.relu(x_t)
                            x_t = self.dropout_layer(x_t)
                        # Pool over nodes: (batch, n_nodes, hidden) -> (batch, hidden)
                        x_t = x_t.mean(dim=1)  # Global average pooling over electrodes
                        time_features.append(x_t)
                    
                    # Stack: (batch, n_time, hidden)
                    x = torch.stack(time_features, dim=1)
                
                # Apply LSTM for temporal modeling
                lstm_out, _ = self.temporal_lstm(x)  # (batch, n_time, hidden*2)
                # Global average pooling over time
                x = lstm_out.mean(dim=1)  # (batch, hidden*2)
                
                # Final classification layers
                x = self.dropout_layer(x)
                x = self.relu(self.fc1(x))
                x = self.dropout_layer(x)
                x = self.fc2(x)
                
                return x
        
        return GCN(input_shape, n_classes, adjacency_matrix, self.gcn_hidden, self.dropout)
    
    def fit(self, X, y, electrode_coordinates=None):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # GNN_STFT_AXIS_FIX: Ensure STFT is (batch, electrodes, freqs, timebins)
        # Incoming from preprocess_stft is (batch, electrodes, timebins, freqs).
        if GNN_FIX_STFT_AXIS_ORDER and X.ndim == 4:
            X = X.transpose(2, 3).contiguous()
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Build adjacency matrix from electrode coordinates
        if electrode_coordinates is None:
            if self.electrode_coordinates is None:
                raise ValueError("electrode_coordinates must be provided either in fit() or set beforehand")
            electrode_coordinates = self.electrode_coordinates
        else:
            self.electrode_coordinates = torch.FloatTensor(electrode_coordinates)
            electrode_coordinates = self.electrode_coordinates
        
        # Build adjacency matrix
        self.adjacency_matrix = self._build_adjacency_matrix(electrode_coordinates)
        self.adjacency_matrix = self.adjacency_matrix.to(self.device)
        
        # Create train/val split
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes, self.adjacency_matrix)
        self.model = self.model.to(self.device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                val_auroc = roc_auc_score(y_val_onehot, val_probs, average='macro', multi_class='ovr')
                
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    log(f"Early stopping at epoch {epoch+1}", priority=3, indent=2)
                    break
            
            if (epoch + 1) % 10 == 0:
                log(f"Epoch {epoch+1}/{self.max_iter}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_auroc={val_auroc:.4f}", priority=3, indent=2)
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self.best_val_auroc = best_val_auroc
    
    def predict_proba(self, X):
        self.model.eval()
        X = torch.FloatTensor(X)

        # GNN_STFT_AXIS_FIX: Keep axis convention consistent with training
        if GNN_FIX_STFT_AXIS_ORDER and X.ndim == 4:
            X = X.transpose(2, 3).contiguous()
        
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class DenoisingAutoencoderClassifier:
    """
    Denoising Autoencoder classifier that learns robust representations by reconstructing
    noisy inputs and uses the encoder output for classification.
    """
    def __init__(self, random_state=42, max_iter=100, batch_size=64, learning_rate=0.001, 
                 val_size=0.2, tol=1e-4, patience=10, noise_level=0.1, 
                 latent_dim=128, recon_weight=0.5, encoder_channels=[32, 64, 128]):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.noise_level = noise_level
        self.latent_dim = latent_dim
        self.recon_weight = recon_weight  # Weight for reconstruction loss vs classification loss
        self.encoder_channels = encoder_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        
    def _add_noise(self, x):
        """Add Gaussian noise to input"""
        noise = torch.randn_like(x) * self.noise_level
        return x + noise
    
    def _create_model(self, input_shape, n_classes):
        class DenoisingAE(torch.nn.Module):
            def __init__(self, input_shape, n_classes, latent_dim=128, 
                        encoder_channels=[32, 64, 128], noise_level=0.1, recon_weight=0.5):
                super().__init__()
                self.latent_dim = latent_dim
                self.recon_weight = recon_weight
                self.noise_level = noise_level
                
                if len(input_shape) == 2:
                    # 2D input: (channels, time)
                    n_channels, n_time = input_shape
                    
                    # Encoder
                    self.encoder_conv = torch.nn.ModuleList()
                    in_channels = n_channels
                    for out_channels in encoder_channels:
                        self.encoder_conv.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
                        in_channels = out_channels
                    
                    # Calculate flattened size after encoder
                    # Do a dummy forward to get the size and shape
                    dummy_input = torch.zeros(1, n_channels, n_time)
                    dummy_out = dummy_input
                    for conv in self.encoder_conv:
                        dummy_out = torch.nn.functional.relu(conv(dummy_out))
                        # Use adaptive pooling if dimension is too small
                        if dummy_out.size(2) > 2:
                            dummy_out = torch.nn.functional.max_pool1d(dummy_out, 2)
                        else:
                            dummy_out = torch.nn.functional.adaptive_avg_pool1d(dummy_out, 1)
                    encoder_output_size = dummy_out.numel()
                    encoder_output_shape = dummy_out.shape[1:]  # (channels, length)
                    
                    self.encoder_fc = torch.nn.Linear(encoder_output_size, latent_dim)
                    
                    # Decoder
                    self.decoder_fc = torch.nn.Linear(latent_dim, encoder_output_size)
                    self.decoder_conv = torch.nn.ModuleList()
                    decoder_channels = list(reversed(encoder_channels))
                    for i in range(len(decoder_channels) - 1):
                        self.decoder_conv.append(torch.nn.ConvTranspose1d(decoder_channels[i], decoder_channels[i+1], 
                                                                          kernel_size=3, stride=2, padding=1, output_padding=1))
                    self.decoder_conv.append(torch.nn.ConvTranspose1d(decoder_channels[-1], n_channels, 
                                                                      kernel_size=3, stride=2, padding=1, output_padding=1))
                    
                    # Classifier head
                    self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(latent_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.5),
                        torch.nn.Linear(256, n_classes)
                    )
                    
                    self.encoder_output_size = encoder_output_size
                    self.encoder_output_shape = encoder_output_shape
                    self.n_time = n_time
                    self.n_channels = n_channels
                    self.encoder_channels = encoder_channels
                    
                else:
                    # 3D input: (channels, freq, time)
                    n_channels, n_freq, n_time = input_shape
                    
                    # Encoder
                    self.encoder_conv = torch.nn.ModuleList()
                    in_channels = n_channels
                    for out_channels in encoder_channels:
                        self.encoder_conv.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                        in_channels = out_channels
                    
                    # Calculate flattened size after encoder
                    dummy_input = torch.zeros(1, n_channels, n_freq, n_time)
                    dummy_out = dummy_input
                    for conv in self.encoder_conv:
                        dummy_out = torch.nn.functional.relu(conv(dummy_out))
                        # Use adaptive pooling if dimensions are too small
                        if dummy_out.size(2) > 2 and dummy_out.size(3) > 2:
                            dummy_out = torch.nn.functional.max_pool2d(dummy_out, 2)
                        elif dummy_out.size(2) == 1 and dummy_out.size(3) > 2:
                            dummy_out = torch.nn.functional.max_pool2d(dummy_out, kernel_size=(1, 2), stride=(1, 2))
                        elif dummy_out.size(2) > 2 and dummy_out.size(3) == 1:
                            dummy_out = torch.nn.functional.max_pool2d(dummy_out, kernel_size=(2, 1), stride=(2, 1))
                        else:
                            dummy_out = torch.nn.functional.adaptive_avg_pool2d(dummy_out, (1, max(1, dummy_out.size(3))))
                    encoder_output_size = dummy_out.numel()
                    encoder_output_shape = dummy_out.shape[1:]  # (channels, freq, time)
                    
                    self.encoder_fc = torch.nn.Linear(encoder_output_size, latent_dim)
                    
                    # Decoder
                    self.decoder_fc = torch.nn.Linear(latent_dim, encoder_output_size)
                    self.decoder_conv = torch.nn.ModuleList()
                    decoder_channels = list(reversed(encoder_channels))
                    for i in range(len(decoder_channels) - 1):
                        self.decoder_conv.append(torch.nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], 
                                                                         kernel_size=3, stride=2, padding=1, output_padding=1))
                    self.decoder_conv.append(torch.nn.ConvTranspose2d(decoder_channels[-1], n_channels, 
                                                                      kernel_size=3, stride=2, padding=1, output_padding=1))
                    
                    # Classifier head
                    self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(latent_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.5),
                        torch.nn.Linear(256, n_classes)
                    )
                    
                    self.encoder_output_size = encoder_output_size
                    self.encoder_output_shape = encoder_output_shape
                    self.n_time = n_time
                    self.n_freq = n_freq
                    self.n_channels = n_channels
                    self.encoder_channels = encoder_channels
                
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(0.5)
            
            def encode(self, x):
                """Encode input to latent representation"""
                for conv in self.encoder_conv:
                    x = self.relu(conv(x))
                    if len(x.shape) == 3:  # 1D
                        # Use adaptive pooling if dimension is too small
                        if x.size(2) > 2:
                            x = torch.nn.functional.max_pool1d(x, 2)
                        else:
                            x = torch.nn.functional.adaptive_avg_pool1d(x, 1)
                    else:  # 2D
                        # Use adaptive pooling if dimensions are too small
                        if x.size(2) > 2 and x.size(3) > 2:
                            x = torch.nn.functional.max_pool2d(x, 2)
                        elif x.size(2) == 1 and x.size(3) > 2:
                            x = torch.nn.functional.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
                        elif x.size(2) > 2 and x.size(3) == 1:
                            x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))
                        else:
                            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, max(1, x.size(3))))
                
                x = x.view(x.size(0), -1)
                z = self.encoder_fc(x)
                return z
            
            def decode(self, z):
                """Decode latent representation to reconstruction"""
                x = self.relu(self.decoder_fc(z))
                
                # Reshape to match encoder output shape (use stored shape)
                batch_size = z.size(0)
                if len(self.encoder_output_shape) == 2:  # 1D: (channels, length)
                    x = x.view(batch_size, *self.encoder_output_shape)
                else:  # 2D: (channels, freq, time)
                    x = x.view(batch_size, *self.encoder_output_shape)
                
                for i, deconv in enumerate(self.decoder_conv):
                    x = self.relu(deconv(x))
                    # Adjust size if needed
                    if len(x.shape) == 3:  # 1D
                        if i < len(self.decoder_conv) - 1:
                            # Try to match expected size
                            target_size = self.n_time // (2 ** (len(self.encoder_channels) - i - 1))
                            if x.size(2) != target_size:
                                x = torch.nn.functional.interpolate(x, size=target_size, mode='linear', align_corners=False)
                    else:  # 2D
                        if i < len(self.decoder_conv) - 1:
                            target_h = self.n_freq // (2 ** (len(self.encoder_channels) - i - 1))
                            target_w = self.n_time // (2 ** (len(self.encoder_channels) - i - 1))
                            if x.size(2) != target_h or x.size(3) != target_w:
                                x = torch.nn.functional.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                # Final adjustment to match input size
                if len(x.shape) == 3:  # 1D
                    if x.size(2) != self.n_time:
                        x = torch.nn.functional.interpolate(x, size=self.n_time, mode='linear', align_corners=False)
                else:  # 2D
                    if x.size(2) != self.n_freq or x.size(3) != self.n_time:
                        x = torch.nn.functional.interpolate(x, size=(self.n_freq, self.n_time), mode='bilinear', align_corners=False)
                
                return x
            
            def forward(self, x, add_noise=False):
                """Forward pass"""
                if add_noise:
                    noise = torch.randn_like(x) * self.noise_level
                    x_noisy = x + noise
                else:
                    x_noisy = x
                
                z = self.encode(x_noisy)
                x_recon = self.decode(z)
                y_pred = self.classifier(z)
                
                return x_recon, y_pred, z
        
        return DenoisingAE(input_shape, n_classes, latent_dim=self.latent_dim, 
                          encoder_channels=self.encoder_channels, noise_level=self.noise_level,
                          recon_weight=self.recon_weight)
    
    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # DAE_STFT_AXIS_FIX: Ensure STFT is (batch, electrodes, freqs, timebins)
        # Incoming from preprocess_stft is (batch, electrodes, timebins, freqs).
        if DAE_FIX_STFT_AXIS_ORDER and X.ndim == 4:
            X = X.transpose(2, 3).contiguous()
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create train/val split
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)
        
        # Setup training
        recon_criterion = torch.nn.MSELoss()
        class_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_recon_loss = 0.0
            train_class_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                x_recon, y_pred, _ = self.model(batch_X, add_noise=True)
                
                recon_loss = recon_criterion(x_recon, batch_X)
                class_loss = class_criterion(y_pred, batch_y)
                loss = self.recon_weight * recon_loss + (1 - self.recon_weight) * class_loss
                
                loss.backward()
                optimizer.step()
                
                train_recon_loss += recon_loss.item() * batch_X.size(0)
                train_class_loss += class_loss.item() * batch_X.size(0)
                _, predicted = torch.max(y_pred.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_recon_loss = train_recon_loss / train_total
            train_class_loss = train_class_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_recon_loss = 0.0
            val_class_loss = 0.0
            
            with torch.no_grad():
                val_outputs_list = []
                for i in range(0, len(X_val), self.batch_size):
                    batch_X = X_val[i:i+self.batch_size].to(self.device)
                    batch_y = y_val[i:i+self.batch_size].to(self.device)
                    
                    x_recon, y_pred, _ = self.model(batch_X, add_noise=False)
                    recon_loss = recon_criterion(x_recon, batch_X)
                    class_loss = class_criterion(y_pred, batch_y)
                    
                    val_recon_loss += recon_loss.item() * batch_X.size(0)
                    val_class_loss += class_loss.item() * batch_X.size(0)
                    val_outputs_list.append(y_pred)
                
                val_outputs = torch.cat(val_outputs_list, dim=0)
                val_recon_loss = val_recon_loss / len(X_val)
                val_class_loss = val_class_loss / len(X_val)
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                log(f"Epoch {epoch+1}/{self.max_iter}: Train recon: {train_recon_loss:.4f}, Train class: {train_class_loss:.4f}, Train acc: {train_acc:.4f}, Val recon: {val_recon_loss:.4f}, Val class: {val_class_loss:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                        break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)

            # DAE_STFT_AXIS_FIX: Keep axis convention consistent with training
            if DAE_FIX_STFT_AXIS_ORDER and X.ndim == 4:
                X = X.transpose(2, 3).contiguous()

            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                _, y_pred, _ = self.model(batch_X, add_noise=False)
                probs = torch.nn.functional.softmax(y_pred, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class VariationalAutoencoderClassifier:
    """
    Variational Autoencoder classifier that learns a probabilistic latent representation
    and uses it for classification.
    """
    def __init__(self, random_state=42, max_iter=100, batch_size=64, learning_rate=0.001, 
                 val_size=0.2, tol=1e-4, patience=10, latent_dim=128, beta=0.01,
                 recon_weight=0.5, encoder_channels=[32, 64, 128]):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.latent_dim = latent_dim
        self.beta = beta  # Weight for KL divergence
        self.recon_weight = recon_weight  # Weight for reconstruction loss vs classification loss
        self.encoder_channels = encoder_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        
    def _create_model(self, input_shape, n_classes):
        class VAE(torch.nn.Module):
            def __init__(self, input_shape, n_classes, latent_dim=128, 
                        encoder_channels=[32, 64, 128], beta=0.01, recon_weight=0.5):
                super().__init__()
                self.latent_dim = latent_dim
                self.beta = beta
                self.recon_weight = recon_weight
                
                if len(input_shape) == 2:
                    # 2D input: (channels, time)
                    n_channels, n_time = input_shape
                    
                    # Encoder
                    self.encoder_conv = torch.nn.ModuleList()
                    in_channels = n_channels
                    for out_channels in encoder_channels:
                        self.encoder_conv.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
                        in_channels = out_channels
                    
                    # Calculate flattened size after encoder
                    dummy_input = torch.zeros(1, n_channels, n_time)
                    dummy_out = dummy_input
                    for conv in self.encoder_conv:
                        dummy_out = torch.nn.functional.relu(conv(dummy_out))
                        # Use adaptive pooling if dimension is too small
                        if dummy_out.size(2) > 2:
                            dummy_out = torch.nn.functional.max_pool1d(dummy_out, 2)
                        else:
                            dummy_out = torch.nn.functional.adaptive_avg_pool1d(dummy_out, 1)
                    encoder_output_size = dummy_out.numel()
                    encoder_output_shape = dummy_out.shape[1:]  # (channels, length)
                    
                    # VAE: mean and logvar layers
                    self.encoder_fc_mean = torch.nn.Linear(encoder_output_size, latent_dim)
                    self.encoder_fc_logvar = torch.nn.Linear(encoder_output_size, latent_dim)
                    
                    # Decoder
                    self.decoder_fc = torch.nn.Linear(latent_dim, encoder_output_size)
                    self.decoder_conv = torch.nn.ModuleList()
                    decoder_channels = list(reversed(encoder_channels))
                    for i in range(len(decoder_channels) - 1):
                        self.decoder_conv.append(torch.nn.ConvTranspose1d(decoder_channels[i], decoder_channels[i+1], 
                                                                          kernel_size=3, stride=2, padding=1, output_padding=1))
                    self.decoder_conv.append(torch.nn.ConvTranspose1d(decoder_channels[-1], n_channels, 
                                                                      kernel_size=3, stride=2, padding=1, output_padding=1))
                    
                    # Classifier head
                    self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(latent_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.5),
                        torch.nn.Linear(256, n_classes)
                    )
                    
                    self.encoder_output_size = encoder_output_size
                    self.encoder_output_shape = encoder_output_shape
                    self.n_time = n_time
                    self.n_channels = n_channels
                    self.encoder_channels = encoder_channels
                    
                else:
                    # 3D input: (channels, freq, time)
                    n_channels, n_freq, n_time = input_shape
                    
                    # Encoder
                    self.encoder_conv = torch.nn.ModuleList()
                    in_channels = n_channels
                    for out_channels in encoder_channels:
                        self.encoder_conv.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                        in_channels = out_channels
                    
                    # Calculate flattened size after encoder
                    dummy_input = torch.zeros(1, n_channels, n_freq, n_time)
                    dummy_out = dummy_input
                    for conv in self.encoder_conv:
                        dummy_out = torch.nn.functional.relu(conv(dummy_out))
                        # Use adaptive pooling if dimensions are too small
                        if dummy_out.size(2) > 2 and dummy_out.size(3) > 2:
                            dummy_out = torch.nn.functional.max_pool2d(dummy_out, 2)
                        elif dummy_out.size(2) == 1 and dummy_out.size(3) > 2:
                            dummy_out = torch.nn.functional.max_pool2d(dummy_out, kernel_size=(1, 2), stride=(1, 2))
                        elif dummy_out.size(2) > 2 and dummy_out.size(3) == 1:
                            dummy_out = torch.nn.functional.max_pool2d(dummy_out, kernel_size=(2, 1), stride=(2, 1))
                        else:
                            dummy_out = torch.nn.functional.adaptive_avg_pool2d(dummy_out, (1, max(1, dummy_out.size(3))))
                    encoder_output_size = dummy_out.numel()
                    encoder_output_shape = dummy_out.shape[1:]  # (channels, freq, time)
                    
                    # VAE: mean and logvar layers
                    self.encoder_fc_mean = torch.nn.Linear(encoder_output_size, latent_dim)
                    self.encoder_fc_logvar = torch.nn.Linear(encoder_output_size, latent_dim)
                    
                    # Decoder
                    self.decoder_fc = torch.nn.Linear(latent_dim, encoder_output_size)
                    self.decoder_conv = torch.nn.ModuleList()
                    decoder_channels = list(reversed(encoder_channels))
                    for i in range(len(decoder_channels) - 1):
                        self.decoder_conv.append(torch.nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], 
                                                                         kernel_size=3, stride=2, padding=1, output_padding=1))
                    self.decoder_conv.append(torch.nn.ConvTranspose2d(decoder_channels[-1], n_channels, 
                                                                      kernel_size=3, stride=2, padding=1, output_padding=1))
                    
                    # Classifier head
                    self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(latent_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.5),
                        torch.nn.Linear(256, n_classes)
                    )
                    
                    self.encoder_output_size = encoder_output_size
                    self.encoder_output_shape = encoder_output_shape
                    self.n_time = n_time
                    self.n_freq = n_freq
                    self.n_channels = n_channels
                    self.encoder_channels = encoder_channels
                
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(0.5)
            
            def encode(self, x):
                """Encode input to latent distribution parameters"""
                for conv in self.encoder_conv:
                    x = self.relu(conv(x))
                    if len(x.shape) == 3:  # 1D
                        # Use adaptive pooling if dimension is too small
                        if x.size(2) > 2:
                            x = torch.nn.functional.max_pool1d(x, 2)
                        else:
                            x = torch.nn.functional.adaptive_avg_pool1d(x, 1)
                    else:  # 2D
                        # Use adaptive pooling if dimensions are too small
                        if x.size(2) > 2 and x.size(3) > 2:
                            x = torch.nn.functional.max_pool2d(x, 2)
                        elif x.size(2) == 1 and x.size(3) > 2:
                            x = torch.nn.functional.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
                        elif x.size(2) > 2 and x.size(3) == 1:
                            x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))
                        else:
                            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, max(1, x.size(3))))
                
                x = x.view(x.size(0), -1)
                mu = self.encoder_fc_mean(x)
                logvar = self.encoder_fc_logvar(x)
                return mu, logvar
            
            def reparameterize(self, mu, logvar):
                """Reparameterization trick"""
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                """Decode latent representation to reconstruction"""
                x = self.relu(self.decoder_fc(z))
                
                # Reshape to match encoder output shape (use stored shape)
                batch_size = z.size(0)
                if len(self.encoder_output_shape) == 2:  # 1D: (channels, length)
                    x = x.view(batch_size, *self.encoder_output_shape)
                else:  # 2D: (channels, freq, time)
                    x = x.view(batch_size, *self.encoder_output_shape)
                
                for i, deconv in enumerate(self.decoder_conv):
                    x = self.relu(deconv(x))
                    # Adjust size if needed
                    if len(x.shape) == 3:  # 1D
                        if i < len(self.decoder_conv) - 1:
                            target_size = self.n_time // (2 ** (len(self.encoder_channels) - i - 1))
                            if x.size(2) != target_size:
                                x = torch.nn.functional.interpolate(x, size=target_size, mode='linear', align_corners=False)
                    else:  # 2D
                        if i < len(self.decoder_conv) - 1:
                            target_h = self.n_freq // (2 ** (len(self.encoder_channels) - i - 1))
                            target_w = self.n_time // (2 ** (len(self.encoder_channels) - i - 1))
                            if x.size(2) != target_h or x.size(3) != target_w:
                                x = torch.nn.functional.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                # Final adjustment to match input size
                if len(x.shape) == 3:  # 1D
                    if x.size(2) != self.n_time:
                        x = torch.nn.functional.interpolate(x, size=self.n_time, mode='linear', align_corners=False)
                else:  # 2D
                    if x.size(2) != self.n_freq or x.size(3) != self.n_time:
                        x = torch.nn.functional.interpolate(x, size=(self.n_freq, self.n_time), mode='bilinear', align_corners=False)
                
                return x
            
            def forward(self, x):
                """Forward pass"""
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                x_recon = self.decode(z)
                y_pred = self.classifier(z)
                
                return x_recon, y_pred, mu, logvar, z
        
        return VAE(input_shape, n_classes, latent_dim=self.latent_dim, 
                  encoder_channels=self.encoder_channels, beta=self.beta,
                  recon_weight=self.recon_weight)
    
    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create train/val split
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)
        
        # Setup training
        recon_criterion = torch.nn.MSELoss()
        class_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_class_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                x_recon, y_pred, mu, logvar, _ = self.model(batch_X)
                
                recon_loss = recon_criterion(x_recon, batch_X)
                # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                class_loss = class_criterion(y_pred, batch_y)
                
                # Combined loss
                loss = self.recon_weight * recon_loss + self.beta * kl_loss + (1 - self.recon_weight) * class_loss
                
                loss.backward()
                optimizer.step()
                
                train_recon_loss += recon_loss.item() * batch_X.size(0)
                train_kl_loss += kl_loss.item() * batch_X.size(0)
                train_class_loss += class_loss.item() * batch_X.size(0)
                _, predicted = torch.max(y_pred.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_recon_loss = train_recon_loss / train_total
            train_kl_loss = train_kl_loss / train_total
            train_class_loss = train_class_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_class_loss = 0.0
            
            with torch.no_grad():
                val_outputs_list = []
                for i in range(0, len(X_val), self.batch_size):
                    batch_X = X_val[i:i+self.batch_size].to(self.device)
                    batch_y = y_val[i:i+self.batch_size].to(self.device)
                    
                    x_recon, y_pred, mu, logvar, _ = self.model(batch_X)
                    recon_loss = recon_criterion(x_recon, batch_X)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    class_loss = class_criterion(y_pred, batch_y)
                    
                    val_recon_loss += recon_loss.item() * batch_X.size(0)
                    val_kl_loss += kl_loss.item() * batch_X.size(0)
                    val_class_loss += class_loss.item() * batch_X.size(0)
                    val_outputs_list.append(y_pred)
                
                val_outputs = torch.cat(val_outputs_list, dim=0)
                val_recon_loss = val_recon_loss / len(X_val)
                val_kl_loss = val_kl_loss / len(X_val)
                val_class_loss = val_class_loss / len(X_val)
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                log(f"Epoch {epoch+1}/{self.max_iter}: Train recon: {train_recon_loss:.4f}, Train KL: {train_kl_loss:.4f}, Train class: {train_class_loss:.4f}, Train acc: {train_acc:.4f}, Val recon: {val_recon_loss:.4f}, Val KL: {val_kl_loss:.4f}, Val class: {val_class_loss:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                        break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size].to(self.device)
                _, y_pred, _, _, _ = self.model(batch_X)
                probs = torch.nn.functional.softmax(y_pred, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class BrainBERTClassifier:
    """
    BrainBERT classifier that uses pretrained or untrained BrainBERT model to extract embeddings,
    then trains a linear classifier on top of the embeddings.
    """
    def __init__(self, random_state=42, brainbert_path=None, pretrained=True, frozen=True, 
                 batch_size=32, max_iter=1000, tol=1e-3):
        """
        Args:
            random_state: Random seed
            brainbert_path: Path to BrainBERT directory. If None, tries to find it relative to current directory.
            pretrained: If True, loads pretrained weights. If False, uses randomly initialized model.
            frozen: If True, freezes BrainBERT weights (only trains linear classifier on top)
            batch_size: Batch size for processing through BrainBERT
            max_iter: Maximum iterations for logistic regression
            tol: Tolerance for logistic regression
        """
        self.random_state = random_state
        self.brainbert_path = brainbert_path
        self.pretrained = pretrained
        self.frozen = frozen
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brainbert_model = None
        self.linear_classifier = None
        self.classes_ = None
        
        # Try to find BrainBERT path if not provided
        if self.brainbert_path is None:
            import os
            # Try common locations
            possible_paths = [
                '../braintree/BrainBERT',
                '../BrainBERT',
                './BrainBERT',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'braintree', 'BrainBERT'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BrainBERT'),
            ]
            for path in possible_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    self.brainbert_path = os.path.abspath(path)
                    break
            
            if self.brainbert_path is None:
                raise ValueError("Could not find BrainBERT directory. Please specify brainbert_path.")
        
        self._load_brainbert_model()
    
    def _load_brainbert_model(self):
        """Load BrainBERT model following demo.ipynb pattern."""
        import os
        import sys
        import torch
        from omegaconf import OmegaConf
        
        # Add BrainBERT to path
        if self.brainbert_path not in sys.path:
            sys.path.insert(0, self.brainbert_path)
        
        # Import models (following demo.ipynb)
        import models
        
        # Define functions from demo.ipynb
        def build_model(cfg):
            ckpt_path = cfg.upstream_ckpt
            init_state = torch.load(ckpt_path, map_location='cpu')
            upstream_cfg = init_state["model_cfg"]
            upstream = models.build_model(upstream_cfg)
            return upstream
        
        def load_model_weights(model, states, multi_gpu):
            if multi_gpu:
                model.module.load_weights(states)
            else:
                model.load_weights(states)
        
        # Find checkpoint
        checkpoint_paths = [
            os.path.join(self.brainbert_path, 'pretrained_weights', 'stft_large_pretrained.pth'),
            os.path.join(self.brainbert_path, 'checkpoints', 'checkpoint_best.pt'),
            os.path.join(self.brainbert_path, 'checkpoint_best.pt'),
        ]
        
        checkpoint_path = None
        for cp_path in checkpoint_paths:
            if os.path.exists(cp_path):
                checkpoint_path = cp_path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"Could not find BrainBERT checkpoint. Tried: {checkpoint_paths}")
        
        # Load model (following demo.ipynb exactly)
        cfg = OmegaConf.create({"upstream_ckpt": checkpoint_path})
        self.brainbert_model = build_model(cfg)
        self.brainbert_model = self.brainbert_model.to(self.device)
        
        if self.pretrained:
            init_state = torch.load(checkpoint_path, map_location='cpu')
            load_model_weights(self.brainbert_model, init_state['model'], False)
        
        if self.frozen:
            for param in self.brainbert_model.parameters():
                param.requires_grad = False
        
        self.brainbert_model.eval()
        log(f"BrainBERT model loaded (pretrained={self.pretrained}, frozen={self.frozen})", priority=2)
    
    def _extract_embeddings(self, X):
        """
        Extract embeddings from neural data using BrainBERT.
        X: np.ndarray of shape (n_samples, n_electrodes, n_timebins, n_freqs) - STFT spectrogram
        Returns: np.ndarray of shape (n_samples, embedding_dim)
        """
        self.brainbert_model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            # Process each sample (following demo.ipynb pattern)
            for i in range(len(X)):
                sample = X[i]  # [electrodes, time, freq]
                
                # Average across electrodes: [time, freq]
                sample = sample.mean(axis=0)
                
                # Convert to tensor and add batch dim: [1, time, freq] (following demo.ipynb)
                inputs = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
                
                # Create mask (all False = no masking, following demo.ipynb)
                mask = torch.zeros(inputs.shape[:2]).bool().to(self.device)
                
                # Forward pass (following demo.ipynb exactly)
                out = self.brainbert_model.forward(inputs, mask, intermediate_rep=True)
                
                # Average over time to get single embedding vector
                embeddings = out.mean(dim=1).cpu().numpy()  # [1, hidden_dim]
                all_embeddings.append(embeddings.squeeze(0))  # [hidden_dim]
        
        return np.array(all_embeddings)


    def _extract_embeddings(self, X):
        """
        Extract embeddings from neural data using BrainBERT.
        X: np.ndarray of shape (n_samples, n_electrodes, n_timebins, n_freqs) - STFT spectrogram
        Returns: np.ndarray of shape (n_samples, embedding_dim)
        """
        self.brainbert_model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(len(X)):
                sample = X[i]  # [electrodes, time, freq]
                
                # Average across electrodes: [time, freq]
                sample = sample.mean(axis=0)  # [time, freq]
                
                # Ensure we have exactly 40 frequency bins (BrainBERT requirement)
                n_freqs = sample.shape[1]
                if n_freqs != 40:
                    if n_freqs < 40:
                        # Pad with zeros if we have fewer than 40 bins
                        padding = np.zeros((sample.shape[0], 40 - n_freqs))
                        sample = np.concatenate([sample, padding], axis=1)
                    else:
                        # Interpolate or slice if we have more than 40 bins
                        from scipy.interpolate import interp1d
                        # Interpolate to exactly 40 bins
                        old_freqs = np.linspace(0, 1, n_freqs)
                        new_freqs = np.linspace(0, 1, 40)
                        sample_interp = np.zeros((sample.shape[0], 40))
                        for t in range(sample.shape[0]):
                            f = interp1d(old_freqs, sample[t, :], kind='linear')
                            sample_interp[t, :] = f(new_freqs)
                        sample = sample_interp
                
                # Convert to tensor and add batch dim: [1, time, freq] (following demo.ipynb)
                inputs = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
                
                # Create mask (all False = no masking, following demo.ipynb)
                mask = torch.zeros(inputs.shape[:2]).bool().to(self.device)
                
                # Forward pass (following demo.ipynb exactly)
                out = self.brainbert_model.forward(inputs, mask, intermediate_rep=True)
                
                # Average over time to get single embedding vector
                embeddings = out.mean(dim=1).cpu().numpy()  # [1, hidden_dim]
                all_embeddings.append(embeddings.squeeze(0))  # [hidden_dim]
        
        return np.array(all_embeddings)
    
    def fit(self, X, y):
        """
        Fit the classifier.
        X: np.ndarray of shape (n_samples, n_electrodes, n_timebins, n_freqs) - STFT spectrogram
        y: np.ndarray of shape (n_samples,) - labels
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        log(f"Extracting BrainBERT embeddings...", priority=2, indent=1)
        
        # Extract embeddings
        X_embeddings = self._extract_embeddings(X)
        
        log(f"Embeddings shape: {X_embeddings.shape}", priority=3, indent=2)
        
        # Standardize embeddings
        scaler = StandardScaler()
        X_embeddings = scaler.fit_transform(X_embeddings)
        self.scaler = scaler
        
        # Train linear classifier
        log(f"Training linear classifier on embeddings...", priority=2, indent=1)
        self.linear_classifier = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol
        )
        self.linear_classifier.fit(X_embeddings, y)
        self.classes_ = self.linear_classifier.classes_
        
        log(f"Training complete.", priority=2, indent=1)
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        # Extract embeddings
        X_embeddings = self._extract_embeddings(X)
        
        # Standardize
        X_embeddings = self.scaler.transform(X_embeddings)
        
        # Predict
        return self.linear_classifier.predict_proba(X_embeddings)
    
    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


############## REGION AVERAGING (FOR DS/DM SPLITS) ###############

def get_region_labels(subject):
    """
    subject: BrainTreebankSubject
    returns: np.ndarray of shape (n_channels,)
    """
    return subject.get_all_electrode_metadata()['DesikanKilliany'].to_numpy()

def combine_regions(X_train, X_test, regions_train, regions_test):
    """
    X_train: np.ndarray of shape (n_samples, n_channels_train, n_timebins, d_model) or (n_samples, n_channels_train, n_timesamples)
    X_test: np.ndarray of shape (n_samples, n_channels_test, n_timebins, d_model) or (n_samples, n_channels_test, n_timesamples)
    regions_train: np.ndarray of shape (n_channels_train,)
    regions_test: np.ndarray of shape (n_channels_test,)
    """
    # Find the intersection of regions between train and test
    unique_regions_train = np.unique(regions_train)
    unique_regions_test = np.unique(regions_test)
    common_regions = np.intersect1d(unique_regions_train, unique_regions_test)
    
    d_model_dimension_unsqueezed = False
    if X_train.ndim == 3:
        # Add a dummy dimension to X_train and X_test for d_model=1
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]
        d_model_dimension_unsqueezed = True

    n_samples_train, _, n_timebins, d_model = X_train.shape
    n_samples_test = X_test.shape[0]
    n_regions_intersect = len(common_regions)
    
    # Create new arrays to store region-averaged data
    X_train_regions = np.zeros((n_samples_train, n_regions_intersect, n_timebins, d_model), dtype=X_train.dtype)
    X_test_regions = np.zeros((n_samples_test, n_regions_intersect, n_timebins, d_model), dtype=X_test.dtype)
    
    # For each common region, average across all channels with that region label
    for i, region in enumerate(common_regions):
        # Find channels corresponding to this region
        train_mask = regions_train == region
        test_mask = regions_test == region
        
        # Average across channels with the same region
        X_train_regions[:, i, :, :] = X_train[:, train_mask, :, :].mean(axis=1)
        X_test_regions[:, i, :, :] = X_test[:, test_mask, :, :].mean(axis=1)

    if d_model_dimension_unsqueezed: # remove the dummy dimension
        X_train_regions = X_train_regions[:, :, :, 0]
        X_test_regions = X_test_regions[:, :, :, 0]
    
    return X_train_regions, X_test_regions, common_regions