import numpy as np
import os
import pickle
import sys
from collections import Counter
from sklearn.model_selection import train_test_split
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import time

# Try to import GPU libraries, but continue if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Add the dataset builder directory to path dynamically
def add_dataset_builder_path(dataset_path):
    if dataset_path and os.path.exists(dataset_path):
        sys.path.append(dataset_path)
        return True
    return False

class GPUManager:
    """Manages available GPUs for efficient resource utilization"""
    def __init__(self):
        self.devices = []
        self.device_memory = {}
        self.current_device_idx = 0
        
        # Check for CUDA GPUs with PyTorch
        if TORCH_AVAILABLE:
            cuda_count = torch.cuda.device_count()
            for i in range(cuda_count):
                device_name = torch.cuda.get_device_name(i)
                mem_info = torch.cuda.get_device_properties(i).total_memory
                self.devices.append(f"cuda:{i}")
                self.device_memory[f"cuda:{i}"] = {
                    'name': device_name,
                    'total_memory': mem_info,
                    'priority': 1 if 'RTX' in device_name else 2  # Prioritize RTX GPUs
                }
                # print(f"Found GPU: {device_name} with {mem_info / 1024**3:.2f} GB memory")
        
        # Sort devices by priority (lower number = higher priority)
        self.devices.sort(key=lambda x: self.device_memory[x]['priority'])
        
        if not self.devices:
            print("No GPU devices found. Using CPU only.")
        # else:
            # print(f"Using GPUs in order: {', '.join([self.device_memory[d]['name'] for d in self.devices])}")
    
    def get_next_device(self):
        """Get the next available device in a round-robin fashion"""
        if not self.devices:
            return None
            
        device = self.devices[self.current_device_idx]
        self.current_device_idx = (self.current_device_idx + 1) % len(self.devices)
        return device

# Initialize the GPU manager globally
gpu_manager = GPUManager()

class TextVectorizer:
    def __init__(self, max_vocab_size=10000, max_sequence_length=100, use_gpu=True):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.use_gpu = use_gpu and (TORCH_AVAILABLE or CUPY_AVAILABLE)
        
        # Determine optimal number of CPU workers
        self.num_workers = max(1, multiprocessing.cpu_count())
        print(f"Using {self.num_workers} CPU workers for parallel processing")
        
    def build_vocabulary(self, texts):
        """Build vocabulary from a list of texts using parallel processing"""
        start_time = time.time()
        # print("Building vocabulary with parallel processing...")
        
        # Process texts in chunks to count words in parallel
        chunk_size = max(1, len(texts) // self.num_workers)
        text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Count words in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._count_words_in_chunk, text_chunks))
        
        # Combine results
        for counter in results:
            self.word_counts.update(counter)
        
        # Select top words for vocabulary
        vocab_words = [word for word, _ in self.word_counts.most_common(self.max_vocab_size-2)]
        
        # Add special tokens
        self.word_to_idx['<PAD>'] = 0  # Padding token
        self.word_to_idx['<UNK>'] = 1  # Unknown token
        
        # Add vocabulary words
        for i, word in enumerate(vocab_words):
            self.word_to_idx[word] = i + 2
        
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary built with {len(self.word_to_idx)} words in {time.time() - start_time:.2f} seconds")
    
    def _count_words_in_chunk(self, text_chunk):
        """Count words in a chunk of texts (used for parallel processing)"""
        counter = Counter()
        for text in text_chunk:
            words = text.lower().split()
            counter.update(words)
        return counter
        
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of indices using parallel processing and GPU acceleration"""
        start_time = time.time()
        
        # Determine batch size based on available memory
        batch_size = min(1000, len(texts))
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        sequences = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Process batch in parallel on CPU
            chunk_size = max(1, len(batch_texts) // self.num_workers)
            text_chunks = [batch_texts[i:i+chunk_size] for i in range(0, len(batch_texts), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_sequences = list(executor.map(self._convert_chunk_to_sequences, text_chunks))
            
            # Flatten the results
            batch_sequences = [seq for chunk in batch_sequences for seq in chunk]
            
            # If GPU is available, pad and optimize on GPU
            if self.use_gpu and batch_sequences:
                device = gpu_manager.get_next_device()
                if device and TORCH_AVAILABLE:
                    sequences.extend(self._pad_sequences_gpu_torch(batch_sequences, device))
                elif CUPY_AVAILABLE:
                    sequences.extend(self._pad_sequences_gpu_cupy(batch_sequences))
                else:
                    sequences.extend(self._pad_sequences_cpu(batch_sequences))
            else:
                sequences.extend(self._pad_sequences_cpu(batch_sequences))
        
        print(f"Converted {len(texts)} texts to sequences in {time.time() - start_time:.2f} seconds")
        return np.array(sequences)
    
    def _convert_chunk_to_sequences(self, text_chunk):
        """Convert a chunk of texts to sequences (without padding)"""
        sequences = []
        for text in text_chunk:
            words = text.lower().split()
            sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
            sequences.append(sequence)
        return sequences
    
    def _pad_sequences_cpu(self, sequences):
        """Pad sequences on CPU"""
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) > self.max_sequence_length:
                padded = sequence[:self.max_sequence_length]
            else:
                padded = sequence + [self.word_to_idx['<PAD>']] * (self.max_sequence_length - len(sequence))
            padded_sequences.append(padded)
        return padded_sequences
    
    def _pad_sequences_gpu_torch(self, sequences, device):
        """Pad sequences using PyTorch GPU acceleration"""
        # Move to GPU for faster processing
        try:
            # First truncate sequences that are too long
            truncated_sequences = [seq[:self.max_sequence_length] if len(seq) > self.max_sequence_length else seq 
                                  for seq in sequences]
            
            # Convert to PyTorch tensors
            tensors = [torch.tensor(seq, dtype=torch.long) for seq in truncated_sequences]
            
            # Pad sequences
            padded = torch.nn.utils.rnn.pad_sequence(
                tensors, 
                batch_first=True, 
                padding_value=self.word_to_idx['<PAD>']
            )
            
            # If sequences are shorter than max_sequence_length, pad more
            if padded.shape[1] < self.max_sequence_length:
                padding = torch.full(
                    (padded.shape[0], self.max_sequence_length - padded.shape[1]), 
                    self.word_to_idx['<PAD>'], 
                    dtype=torch.long
                )
                padded = torch.cat([padded, padding], dim=1)
            
            # Convert back to numpy and return
            return padded.cpu().numpy().tolist()
        except Exception as e:
            print(f"GPU processing failed: {e}. Falling back to CPU processing.")
            return self._pad_sequences_cpu(sequences)
    
    def _pad_sequences_gpu_cupy(self, sequences):
        """Pad sequences using CuPy GPU acceleration"""
        try:
            # First truncate and prepare sequences
            max_len = self.max_sequence_length
            pad_value = self.word_to_idx['<PAD>']
            
            # Prepare arrays for CuPy
            padded_sequences = []
            for seq in sequences:
                if len(seq) > max_len:
                    padded = seq[:max_len]
                else:
                    padded = seq + [pad_value] * (max_len - len(seq))
                padded_sequences.append(padded)
            
            # Convert to CuPy array for potential future GPU operations
            cp_array = cp.array(padded_sequences, dtype=cp.int32)
            
            # Convert back to numpy for return
            return cp.asnumpy(cp_array).tolist()
        except Exception as e:
            print(f"CuPy processing failed: {e}. Falling back to CPU processing.")
            return self._pad_sequences_cpu(sequences)
    
    def save(self, filepath):
        """Save the vectorizer to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'word_counts': self.word_counts,
                'max_vocab_size': self.max_vocab_size,
                'max_sequence_length': self.max_sequence_length
            }, f)
            
    @classmethod
    def load(cls, filepath):
        """Load a vectorizer from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        vectorizer = cls(data['max_vocab_size'], data['max_sequence_length'])
        vectorizer.word_to_idx = data['word_to_idx']
        vectorizer.idx_to_word = data['idx_to_word']
        vectorizer.word_counts = data['word_counts']
        return vectorizer

def prepare_data(data_path=None, dataset_path=None, test_size=0.2, vectorizer=None):
    """Prepare data for training and testing"""
    # Load the dataset
    if data_path is None:
        # First check if we can use our bridge
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        try:
            from dataset_bridge import load_dataset
            texts, labels = load_dataset(dataset_path)
        except (ImportError, Exception) as e:
            print(f"Warning: Could not use dataset_bridge: {e}")
            
            # Fallback to original approach
            if add_dataset_builder_path(dataset_path):
                try:
                    # Check if dataset_builder.py exists in the current directory
                    try:
                        from dataset_builder import load_text_files
                        # Using load_text_files directly since load_dataset expects a file path
                        text_files = load_text_files(dataset_path)
                        if text_files:
                            texts = list(text_files.values())
                            # Simple binary classification based on text length
                            labels = [1 if len(text) > 500 else 0 for text in texts]
                        else:
                            raise ValueError(f"No text files found in {dataset_path}")
                    except (ImportError, AttributeError):
                        # Load text files directly from directory
                        import glob
                        text_files = glob.glob(os.path.join(dataset_path, "*.txt"))
                        if not text_files:
                            text_files = glob.glob(os.path.join(dataset_path, "**", "*.txt"), recursive=True)
                            
                        if not text_files:
                            raise ValueError(f"No text files found in {dataset_path}")
                            
                        texts = []
                        labels = []
                        for i, file_path in enumerate(text_files):
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    text = f.read()
                                texts.append(text)
                                # Simple label based on filename or position
                                filename = os.path.basename(file_path)
                                if "_class_" in filename:
                                    label = int(filename.split("_class_")[1].split("_")[0])
                                else:
                                    # Use directory name as class if possible
                                    parent_dir = os.path.basename(os.path.dirname(file_path))
                                    if parent_dir != os.path.basename(dataset_path):
                                        label = hash(parent_dir) % 2  # Simple binary classification
                                    else:
                                        label = i % 2  # Alternate between 0 and 1
                                labels.append(label)
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                except Exception as e:
                    raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")
            else:
                raise ValueError(f"Dataset path not found: {dataset_path}")
    else:
        # Load from provided data path
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different possible data formats
            if isinstance(data, dict):
                # Case 1: Direct dictionary with 'texts' and 'labels' keys
                if 'texts' in data and 'labels' in data:
                    texts, labels = data['texts'], data['labels']
                    print(f"Loaded {len(texts)} samples directly from {data_path}")
                
                # Case 2: Dictionary with train/test splits already defined
                elif ('x_train' in data or 'X_train' in data) and ('y_train' in data or 'Y_train' in data):
                    # Get the correct case of keys
                    x_train_key = 'x_train' if 'x_train' in data else 'X_train'
                    y_train_key = 'y_train' if 'y_train' in data else 'Y_train'
                    x_test_key = 'x_test' if 'x_test' in data else 'X_test'
                    y_test_key = 'y_test' if 'y_test' in data else 'Y_test'
                    
                    # Check if the data is already vectorized
                    x_train = data[x_train_key]
                    y_train = data[y_train_key]
                    x_test = data[x_test_key]
                    y_test = data[y_test_key]
                    
                    # If data is already vectorized (numerical arrays)
                    if isinstance(x_train, np.ndarray) and x_train.dtype != object:
                        if vectorizer is None:
                            raise ValueError("Data is already vectorized but no vectorizer provided")
                        print(f"Using pre-vectorized data from {data_path}")
                        return {
                            'x_train': x_train,
                            'y_train': np.array(y_train),
                            'x_test': x_test,
                            'y_test': np.array(y_test),
                            'vectorizer': vectorizer
                        }
                    
                    # If data is text, combine train and test for further processing
                    texts = list(x_train) + list(x_test)
                    labels = list(y_train) + list(y_test)
                    print(f"Loaded {len(texts)} samples from pre-split dataset")
                else:
                    # Try to extract text and labels from custom format
                    raise ValueError(f"Could not identify expected keys in data file {data_path}")
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                # Case 3: Tuple/List with (texts, labels)
                texts, labels = data
                print(f"Loaded {len(texts)} samples from tuple data")
            else:
                raise ValueError(f"Unsupported data format in {data_path}")
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_path}: {e}")
    
    # Split data if not already split
    if 'x_train' not in locals():
        x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42)
    
    # Create or use vectorizer
    if vectorizer is None:
        vectorizer = TextVectorizer()
        vectorizer.build_vocabulary(x_train)
    
    # Convert texts to sequences (if not already in sequence form)
    if isinstance(x_train, (list, tuple)) and all(isinstance(item, str) for item in x_train[:10]):
        x_train_seq = vectorizer.texts_to_sequences(x_train)
        x_test_seq = vectorizer.texts_to_sequences(x_test)
    else:
        # Data might already be in sequence form
        x_train_seq = np.array(x_train)
        x_test_seq = np.array(x_test)
    
    return {
        'x_train': x_train_seq,
        'y_train': np.array(y_train),
        'x_test': x_test_seq,
        'y_test': np.array(y_test),
        'vectorizer': vectorizer
    }

def load_direct_data(training_data, testing_data=None, training_labels=None, testing_labels=None, vectorizer=None):
    """
    Load training and testing data from directly specified files
    
    Args:
        training_data (str): Path to training data file
        testing_data (str, optional): Path to testing data file
        training_labels (str, optional): Path to training labels file
        testing_labels (str, optional): Path to testing labels file
        vectorizer (TextVectorizer, optional): Vectorizer to use
    
    Returns:
        dict: Dictionary containing training and testing data
    """
    print(f"Loading data from specified files...")
    
    # Helper function to read text file
    def read_text_file(file_path):
        if not file_path:
            return []
        
        print(f"Reading {file_path}...")
        
        # Determine file type from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pkl':
            # Pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data
        elif ext in ['.npy', '.npz']:
            # NumPy file
            return np.load(file_path)
        else:
            # Text file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Check if this is likely raw text or a list of items
            if len(lines) == 1 and len(lines[0]) > 1000:
                # Probably a single document
                return [lines[0]]
            else:
                # List of items (one per line)
                return [line.strip() for line in lines if line.strip()]
    
    # Load training data
    x_train = read_text_file(training_data)
    
    # Validate training data
    if not x_train:
        raise ValueError("Training data is empty. Please check your training data file.")
    
    print(f"Loaded {len(x_train)} training samples")
    
    # Check data quality
    sample_lengths = [len(text) for text in x_train[:100]]
    avg_length = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
    
    if avg_length < 10:
        print(f"Warning: Training samples appear very short (average length: {avg_length:.1f} chars)")
        print("This may affect model performance. Consider using more complete data.")
    
    # Try to infer if the training data contains labels
    contains_labels = False
    if isinstance(x_train, (list, tuple)) and len(x_train) > 0:
        # Check the first few items to see if they might be texts with labels
        sample_items = x_train[:min(5, len(x_train))]
        # If items are strings with tab separators, they might contain labels
        if all(isinstance(item, str) and '\t' in item for item in sample_items):
            print("Detected tab-separated format in training data, extracting labels...")
            texts_and_labels = [item.split('\t', 1) for item in x_train]
            x_train = [item[1] for item in texts_and_labels if len(item) > 1]
            y_train = [int(item[0]) if item[0].isdigit() else item[0] for item in texts_and_labels if len(item) > 0]
            contains_labels = True
    
    # Load labels if not extracted from training data
    if not contains_labels:
        if training_labels:
            y_train = read_text_file(training_labels)
            # Convert to integers if possible
            try:
                y_train = [int(y) for y in y_train]
            except (ValueError, TypeError):
                pass
        else:
            # For datasets without labels, try to infer classes through clustering
            try:
                # Import here to avoid namespace conflicts with our own TextVectorizer
                from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
                from sklearn.cluster import KMeans
                
                # Try to infer at least 2 classes through clustering if we have enough samples
                if len(x_train) >= 20:
                    print("No labels found. Attempting to infer classes through text clustering...")
                    # Use TF-IDF vectorization for text clustering
                    clustering_vectorizer = SklearnTfidfVectorizer(max_features=1000, stop_words='english')
                    
                    # Use a sample of texts if dataset is large (to save memory)
                    sample_size = min(10000, len(x_train))
                    if sample_size < len(x_train):
                        print(f"Using a sample of {sample_size} texts for clustering (out of {len(x_train)} total)")
                        import random
                        random.seed(42)  # For reproducibility
                        sample_indices = random.sample(range(len(x_train)), sample_size)
                        sampled_texts = [x_train[i][:10000] for i in sample_indices]  # Limit text length too
                        X_tfidf = clustering_vectorizer.fit_transform(sampled_texts)
                    else:
                        X_tfidf = clustering_vectorizer.fit_transform([t[:10000] for t in x_train])  # Limit text length
                    
                    # Determine optimal number of clusters (2-5)
                    from sklearn.metrics import silhouette_score
                    best_score = -1
                    best_n_clusters = 2
                    
                    for n_clusters in range(2, min(6, len(x_train) // 10 + 1)):
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(X_tfidf)
                        if len(set(cluster_labels)) > 1:  # Ensure we have at least 2 clusters
                            score = silhouette_score(X_tfidf, cluster_labels)
                            if score > best_score:
                                best_score = score
                                best_n_clusters = n_clusters
                    
                    # Get cluster assignments for all texts
                    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
                    
                    # If we used sampling, we need to assign clusters to all texts
                    if sample_size < len(x_train):
                        # Train on the sample
                        kmeans.fit(X_tfidf)
                        # Transform all texts
                        all_X_tfidf = clustering_vectorizer.transform([t[:10000] for t in x_train])
                        # Predict clusters for all texts
                        y_train = kmeans.predict(all_X_tfidf).tolist()
                    else:
                        # No sampling, just fit and predict
                        y_train = kmeans.fit_predict(X_tfidf).tolist()
                        
                    print(f"Clustered texts into {best_n_clusters} classes with silhouette score: {best_score:.3f}")
                    
                    # Clean up to free memory
                    del clustering_vectorizer
                    del kmeans
                    del X_tfidf
                    if 'all_X_tfidf' in locals():
                        del all_X_tfidf
                    
                else:
                    # Fallback to simple binary labels
                    print("No labels found and too few samples for clustering. Using alternating labels...")
                    y_train = [i % 2 for i in range(len(x_train))]
            except (ImportError, Exception) as e:
                # If clustering fails, use dummy labels
                print(f"Clustering failed: {e}")
                print("Using dummy labels (all 0).")
                y_train = [0] * len(x_train)
    
    # Load testing data
    if testing_data:
        x_test = read_text_file(testing_data)
        
        # Check if testing data also contains labels
        contains_test_labels = False
        if isinstance(x_test, (list, tuple)) and len(x_test) > 0:
            sample_items = x_test[:min(5, len(x_test))]
            if all(isinstance(item, str) and '\t' in item for item in sample_items):
                print("Detected tab-separated format in testing data, extracting labels...")
                texts_and_labels = [item.split('\t', 1) for item in x_test]
                x_test = [item[1] for item in texts_and_labels if len(item) > 1]
                y_test = [int(item[0]) if item[0].isdigit() else item[0] for item in texts_and_labels if len(item) > 0]
                contains_test_labels = True
        
        # Load testing labels if not extracted
        if not contains_test_labels:
            if testing_labels:
                y_test = read_text_file(testing_labels)
                # Convert to integers if possible
                try:
                    y_test = [int(y) for y in y_test]
                except (ValueError, TypeError):
                    pass
            else:
                # For consistency, use the same approach as training set if possible
                if len(set(y_train)) > 1:
                    print("No testing labels found. Using similar distribution as training labels...")
                    # Sample from the same distribution as training labels
                    from random import choices
                    train_unique_labels = list(set(y_train))
                    train_label_weights = [y_train.count(label)/len(y_train) for label in train_unique_labels]
                    y_test = choices(train_unique_labels, weights=train_label_weights, k=len(x_test))
                else:
                    print("No testing labels found. Using dummy labels (all 0).")
                    y_test = [0] * len(x_test)
    else:
        # Split training data if no testing data
        print("No testing data provided. Splitting training data...")
        split_idx = int(0.8 * len(x_train))
        x_test = x_train[split_idx:]
        y_test = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
    
    print(f"Loaded {len(x_train)} training samples and {len(x_test)} testing samples")
    print(f"Label distribution in training: {dict([(label, y_train.count(label)) for label in set(y_train)])}")
    
    # Create or use vectorizer
    if vectorizer is None:
        vectorizer = TextVectorizer()
        # Check if the data is already vectorized (numerical)
        if isinstance(x_train, np.ndarray) and x_train.dtype != object:
            print("Data appears to be already vectorized.")
        else:
            # Build vocabulary from texts
            # print("Building vocabulary...")
            vectorizer.build_vocabulary(x_train)
    
    # Convert texts to sequences if they are not already
    if isinstance(x_train, (list, tuple)) and isinstance(x_train[0], str):
        print("Converting texts to sequences...")
        x_train_seq = vectorizer.texts_to_sequences(x_train)
        x_test_seq = vectorizer.texts_to_sequences(x_test)
    else:
        # Data might already be in sequence form
        x_train_seq = np.array(x_train)
        x_test_seq = np.array(x_test)
    
    return {
        'x_train': x_train_seq,
        'y_train': np.array(y_train),
        'x_test': x_test_seq,
        'y_test': np.array(y_test),
        'vectorizer': vectorizer
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text vectorization for NLP')
    parser.add_argument('--data_path', type=str, help='Path to preprocessed data file')
    parser.add_argument('--dataset_path', type=str, default='download_and_create_textdata', 
                       help='Path to directory containing dataset_builder.py')
    parser.add_argument('--output_path', type=str, default='models', 
                       help='Path to save vectorizer')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--cpu_workers', type=int, help='Number of CPU workers (default: auto)')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    dataset_path = os.path.abspath(args.dataset_path) if args.dataset_path else None
    output_path = os.path.abspath(args.output_path) if args.output_path else 'models'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Example usage
    if args.cpu_workers:
        # Override default number of workers if specified
        multiprocessing.set_start_method('spawn', force=True)
        
    data = prepare_data(args.data_path, dataset_path, vectorizer=TextVectorizer(use_gpu=args.use_gpu))
    print(f"Train data shape: {data['x_train'].shape}")
    print(f"Test data shape: {data['x_test'].shape}")
    
    # Save the vectorizer
    vectorizer_path = os.path.join(output_path, "vectorizer.pkl")
    data['vectorizer'].save(vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")
