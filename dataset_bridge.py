"""
Bridge module that connects the existing dataset_builder.py functions with 
the expected interface used by text_vectorization.py
"""

import os
import sys
import glob
import random
from typing import List, Tuple, Dict, Any, Union

def load_dataset(dataset_path: str) -> Tuple[List[str], List[int]]:
    """
    Loads text data from the given dataset path.
    
    This function serves as a bridge between the existing dataset_builder.py
    and the expected interface in text_vectorization.py
    
    Args:
        dataset_path: Path to directory containing text files
        
    Returns:
        tuple: (texts, labels) where texts is a list of strings and labels is a list of integers
    """
    # Check if path is a directory or file
    if os.path.isfile(dataset_path):
        # If it's a file, try to load it as a pickle file
        try:
            import pickle
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            # Extract texts and labels if it's our expected format
            if isinstance(dataset, dict):
                texts = []
                labels = []
                
                # Try to concatenate train and test sets if available
                for prefix in ['', 'X_', 'x_']:
                    train_key = f'{prefix}train'
                    test_key = f'{prefix}test'
                    
                    if train_key in dataset:
                        texts.extend(dataset[train_key])
                    if test_key in dataset:
                        texts.extend(dataset[test_key])
                
                # Try to extract labels
                for prefix in ['', 'Y_', 'y_']:
                    train_key = f'{prefix}train'
                    test_key = f'{prefix}test'
                    
                    if train_key in dataset:
                        labels.extend(dataset[train_key])
                    if test_key in dataset:
                        labels.extend(dataset[test_key])
                
                # If we have texts but no labels, create dummy labels
                if texts and not labels:
                    labels = [0] * len(texts)
                
                if texts:
                    print(f"Loaded {len(texts)} samples from pickle file")
                    return texts, labels
        except Exception as e:
            print(f"Failed to load pickle file: {e}")
    
    # Check if the path is a directory
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Path is not a valid directory or file: {dataset_path}")
    
    # Try to import dataset_builder if available
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # Try to use dataset_builder's load_text_files
        from dataset_builder import load_text_files
        
        text_files = load_text_files(dataset_path)
        if text_files:
            texts = list(text_files.values())
            # Create simple labels based on text length
            labels = [1 if len(text) > 500 else 0 for text in texts]
            print(f"Loaded {len(texts)} texts using dataset_builder.load_text_files()")
            return texts, labels
    except (ImportError, Exception) as e:
        print(f"Could not use dataset_builder.load_text_files: {e}")
    
    # Fallback: Load text files directly from the directory
    print(f"Loading text files directly from directory: {dataset_path}")
    texts = []
    labels = []
    
    # First try to find text files in the root directory
    text_files = glob.glob(os.path.join(dataset_path, "*.txt"))
    
    # If no files found, try to search in subdirectories
    if not text_files:
        text_files = glob.glob(os.path.join(dataset_path, "**", "*.txt"), recursive=True)
    
    if not text_files:
        # Last resort - try other extensions
        for ext in ['.text', '.md', '.csv', '.json']:
            text_files = glob.glob(os.path.join(dataset_path, f"*{ext}"), recursive=False)
            if text_files:
                break
                
            text_files = glob.glob(os.path.join(dataset_path, f"**/*{ext}"), recursive=True)
            if text_files:
                break
    
    # Process each text file
    for i, file_path in enumerate(text_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                
            if len(text.strip()) > 0:
                texts.append(text)
                
                # Try to determine a label from the file path
                file_name = os.path.basename(file_path)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                
                # Try to extract label from filename if it contains class information
                if "_class_" in file_name:
                    try:
                        label = int(file_name.split("_class_")[1].split("_")[0])
                    except (ValueError, IndexError):
                        label = i % 2  # Fallback to alternating labels
                # Use parent directory as category if not the root dataset directory
                elif parent_dir != os.path.basename(dataset_path):
                    # Convert directory name to a numeric label (hash modulo 2 for binary)
                    label = hash(parent_dir) % 2
                else:
                    # Use alternating labels (0, 1) as fallback
                    label = i % 2
                
                labels.append(label)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    if not texts:
        print("No text files found. Creating a synthetic dataset.")
        # Create synthetic dataset as last resort
        texts = [f"This is sample text {i} for training." for i in range(100)]
        labels = [random.randint(0, 1) for _ in range(100)]
    
    print(f"Loaded {len(texts)} texts with {len(set(labels))} unique labels")
    return texts, labels

# For testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the dataset bridge")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    
    args = parser.parse_args()
    
    texts, labels = load_dataset(args.dataset_path)
    print(f"Loaded {len(texts)} texts with {len(set(labels))} unique labels")
    
    # Print some samples
    for i in range(min(3, len(texts))):
        print(f"\nSample {i+1} (Label: {labels[i]}):")
        print(texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i])
