"""
Dataset builder for NLP project.

This module provides functions to build training and testing datasets
from cleaned text data, analyze text quality, and visualize metrics.
"""

import os
import re
import sys  # Add this import
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Union, Optional
import textstat
import pickle

# Import from our text cleaning module
from text_cleaning import clean_text, detect_language

def load_text_files(directory: str, file_pattern: str = "*.txt") -> Dict[str, str]:
    """
    Load all text files matching the pattern from a directory
    
    Args:
        directory (str): Directory containing text files
        file_pattern (str): Pattern to match files (default: "*.txt")
        
    Returns:
        dict: Dictionary mapping filenames to text content
    """
    import glob
    
    text_files = {}
    
    for filepath in glob.glob(os.path.join(directory, file_pattern)):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            filename = os.path.basename(filepath)
            text_files[filename] = content
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return text_files

def compute_text_metrics(text: str, lang: str = 'en') -> Dict[str, float]:
    """
    Compute quality metrics for a given text
    
    Args:
        text (str): Text to analyze
        lang (str): Language code
        
    Returns:
        dict: Dictionary of text quality metrics
    """
    if not text or len(text.strip()) == 0:
        return {
            'length': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'vocab_richness': 0,
            'readability': 0,
            'stopword_ratio': 0
        }
    
    # Get word and sentence counts
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    # Vocabulary richness (unique words / total words)
    vocab_richness = len(set(words)) / len(words) if words else 0
    
    # Average lengths
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    avg_sentence_length = sum(len(re.findall(r'\b\w+\b', s.lower())) for s in sentences) / len(sentences) if sentences else 0
    
    # Get readability score based on language
    if lang == 'en':
        readability = textstat.flesch_reading_ease(text)
    else:
        # For non-English, use a simpler metric that works cross-linguistically
        readability = 100 - min(100, 4.71 * avg_word_length + 0.5 * avg_sentence_length)
    
    # Compute stopword ratio if it's a supported language
    stopword_ratio = 0
    try:
        from nltk.corpus import stopwords
        nltk_lang = {'en': 'english', 'es': 'spanish', 'fr': 'french', 
                    'de': 'german', 'it': 'italian', 'pt': 'portuguese',
                    'nl': 'dutch', 'ru': 'russian'}.get(lang, None)
        
        if nltk_lang:
            stop_words = set(stopwords.words(nltk_lang))
            stopword_count = sum(1 for word in words if word in stop_words)
            stopword_ratio = stopword_count / len(words) if words else 0
    except:
        # NLTK stopwords not available, skip this metric
        pass
        
    return {
        'length': len(text),
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'vocab_richness': vocab_richness,
        'readability': readability,
        'stopword_ratio': stopword_ratio
    }

def analyze_dataset_quality(texts: List[str], langs: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze the quality of a text dataset
    
    Args:
        texts (list): List of text samples
        langs (list, optional): List of language codes corresponding to each text
        
    Returns:
        DataFrame: DataFrame containing quality metrics for each text
    """
    if langs is None:
        # Detect languages if not provided
        langs = [detect_language(text) for text in texts]
    
    metrics_list = []
    
    for i, (text, lang) in enumerate(zip(texts, langs)):
        metrics = compute_text_metrics(text, lang)
        metrics['id'] = i
        metrics['language'] = lang
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)

def visualize_text_quality(metrics_df: pd.DataFrame, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Create visualizations of text quality metrics
    
    Args:
        metrics_df (DataFrame): DataFrame containing text quality metrics
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Dictionary mapping visualization names to file paths (if saved)
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    visualization_paths = {}
    
    # Set the style for better-looking plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Text length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['length'], bins=30, alpha=0.7, color='blue')
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, 'text_length_distribution.png')
        plt.savefig(path)
        visualization_paths['text_length'] = path
    plt.close()
    
    # 2. Vocabulary richness vs. text length
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['length'], metrics_df['vocab_richness'], alpha=0.5)
    plt.title('Vocabulary Richness vs. Text Length')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Vocabulary Richness (unique words / total words)')
    plt.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, 'vocab_richness_vs_length.png')
        plt.savefig(path)
        visualization_paths['vocab_richness'] = path
    plt.close()
    
    # 3. Average sentence length by language
    if 'language' in metrics_df.columns:
        plt.figure(figsize=(12, 6))
        avg_by_lang = metrics_df.groupby('language')['avg_sentence_length'].mean().sort_values(ascending=False)
        avg_by_lang.plot(kind='bar', color='teal')
        plt.title('Average Sentence Length by Language')
        plt.xlabel('Language')
        plt.ylabel('Average Words per Sentence')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            path = os.path.join(output_dir, 'avg_sentence_by_language.png')
            plt.savefig(path)
            visualization_paths['avg_sentence_by_lang'] = path
        plt.close()
    
    # 4. Readability scores distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['readability'], bins=30, alpha=0.7, color='green')
    plt.title('Readability Score Distribution')
    plt.xlabel('Readability Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, 'readability_distribution.png')
        plt.savefig(path)
        visualization_paths['readability'] = path
    plt.close()
    
    # 5. Word length vs. sentence length
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['avg_word_length'], metrics_df['avg_sentence_length'], alpha=0.5)
    plt.title('Word Length vs. Sentence Length')
    plt.xlabel('Average Word Length (characters)')
    plt.ylabel('Average Sentence Length (words)')
    plt.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, 'word_vs_sentence_length.png')
        plt.savefig(path)
        visualization_paths['word_vs_sentence'] = path
    plt.close()
    
    # 6. Quality metrics correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ['length', 'avg_word_length', 'avg_sentence_length', 
                   'vocab_richness', 'readability', 'stopword_ratio']
    corr = metrics_df[numeric_cols].corr()
    
    import seaborn as sns
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation between Text Quality Metrics')
    plt.tight_layout()
    
    if output_dir:
        path = os.path.join(output_dir, 'metrics_correlation.png')
        plt.savefig(path)
        visualization_paths['metrics_correlation'] = path
    plt.close()
    
    return visualization_paths

def create_dataset(texts: List[str], 
                  labels: Optional[List[Union[str, int]]] = None,
                  test_size: float = 0.2, 
                  random_state: int = 42,
                  min_length: int = 100,
                  balance_classes: bool = False) -> Dict[str, Union[List, np.ndarray]]:
    """
    Create training and testing datasets from text data
    
    Args:
        texts (list): List of text samples
        labels (list, optional): List of labels for classification tasks
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        min_length (int): Minimum text length to include
        balance_classes (bool): Whether to balance classes in classification tasks
        
    Returns:
        dict: Dictionary containing train and test splits
    """
    # Filter out texts that are too short
    filtered_indices = [i for i, text in enumerate(texts) if len(text) >= min_length]
    filtered_texts = [texts[i] for i in filtered_indices]
    
    if labels is not None:
        filtered_labels = [labels[i] for i in filtered_indices]
        
        if balance_classes:
            # Get counts for each class
            label_counts = Counter(filtered_labels)
            min_count = min(label_counts.values())
            
            # Select equal amounts from each class
            balanced_texts = []
            balanced_labels = []
            
            for label in label_counts:
                indices = [i for i, l in enumerate(filtered_labels) if l == label]
                selected_indices = random.sample(indices, min_count)
                
                balanced_texts.extend([filtered_texts[i] for i in selected_indices])
                balanced_labels.extend([filtered_labels[i] for i in selected_indices])
            
            filtered_texts = balanced_texts
            filtered_labels = balanced_labels
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_texts, filtered_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=filtered_labels
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    else:
        # For unsupervised learning (no labels)
        train_texts, test_texts = train_test_split(
            filtered_texts, 
            test_size=test_size, 
            random_state=random_state
        )
        
        return {
            'X_train': train_texts,
            'X_test': test_texts
        }

def save_dataset(dataset: Dict[str, Union[List, np.ndarray]], 
                output_dir: str, 
                name: str = 'nlp_dataset') -> str:
    """
    Save the dataset to disk
    
    Args:
        dataset (dict): Dataset dictionary with train/test splits
        output_dir (str): Directory to save the dataset
        name (str): Base name for the dataset files
        
    Returns:
        str: Path to the saved dataset
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as pickle for preserving Python objects
    dataset_path = os.path.join(output_dir, f'{name}.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    # Also save text files for human readability
    for key, data in dataset.items():
        if isinstance(data, (list, np.ndarray)):
            txt_path = os.path.join(output_dir, f'{name}_{key}.txt')
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                if isinstance(data[0], str):
                    # For text data, add separators between samples
                    for i, item in enumerate(data):
                        f.write(f"--- Sample {i+1} ---\n")
                        f.write(item)
                        f.write("\n\n")
                else:
                    # For labels or other data
                    for item in data:
                        f.write(f"{item}\n")
    
    # Save a metadata file with information about the dataset
    metadata = {
        'name': name,
        'total_samples': len(dataset.get('X_train', [])) + len(dataset.get('X_test', [])),
        'train_samples': len(dataset.get('X_train', [])),
        'test_samples': len(dataset.get('X_test', [])),
        'has_labels': 'y_train' in dataset,
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(output_dir, f'{name}_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return dataset_path

def load_dataset(dataset_path: str) -> Dict[str, Union[List, np.ndarray]]:
    """
    Load a saved dataset
    
    Args:
        dataset_path (str): Path to the saved dataset
        
    Returns:
        dict: Dataset dictionary with train/test splits
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset

def main(input_dir: str, output_dir: str, **kwargs):
    """
    Main function to process a directory of text files and build a dataset
    
    Args:
        input_dir (str): Directory containing input text files
        output_dir (str): Directory to save output datasets and visualizations
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load and process text files
    text_files = load_text_files(input_dir)
    print(f"Loaded {len(text_files)} text files from {input_dir}")
    
    # Extract texts and detect languages
    texts = list(text_files.values())
    languages = [detect_language(text) for text in texts]
    
    # Analyze text quality
    metrics_df = analyze_dataset_quality(texts, languages)
    metrics_path = os.path.join(output_dir, 'text_quality_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved text quality metrics to {metrics_path}")
    
    # Create visualizations
    vis_paths = visualize_text_quality(metrics_df, vis_dir)
    print(f"Created {len(vis_paths)} visualizations in {vis_dir}")
    
    # Create dataset (unsupervised by default)
    dataset = create_dataset(
        texts, 
        test_size=kwargs.get('test_size', 0.2),
        min_length=kwargs.get('min_length', 100)
    )
    
    # Save dataset
    dataset_path = save_dataset(dataset, output_dir, name=kwargs.get('name', 'nlp_dataset'))
    print(f"Saved dataset to {dataset_path}")
    
    # Summary report
    summary = {
        'total_files': len(text_files),
        'total_samples_after_filtering': len(dataset['X_train']) + len(dataset['X_test']),
        'train_samples': len(dataset['X_train']),
        'test_samples': len(dataset['X_test']),
        'languages': dict(Counter(languages)),
        'avg_text_length': metrics_df['length'].mean(),
        'visualizations': list(vis_paths.keys())
    }
    
    summary_path = os.path.join(output_dir, 'summary_report.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\nDataset creation complete!")
    print(f"Created {summary['train_samples']} training and {summary['test_samples']} testing samples")
    print(f"Summary report saved to {summary_path}")

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Build NLP datasets from cleaned text files.')
    
    # Create argument groups to make input/output arguments optional
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input', '-i', help='Directory containing input text files')
    
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--output', '-o', help='Directory to save output datasets')
    
    # Add optional arguments
    parser.add_argument('--test-size', '-t', type=float, default=0.2, help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--min-length', '-m', type=int, default=100, help='Minimum text length to include (default: 100)')
    parser.add_argument('--name', '-n', default='nlp_dataset', help='Base name for dataset files (default: nlp_dataset)')
    
    # Add positional arguments as alternatives to --input and --output
    # These must be the last arguments defined to ensure proper parsing of optional args
    input_group.add_argument('input_dir_pos', nargs='?', help='Positional directory containing input text files')
    output_group.add_argument('output_dir_pos', nargs='?', help='Positional directory to save output datasets')
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Use named arguments if provided, otherwise use positional arguments
    input_dir = args.input if args.input else args.input_dir_pos
    output_dir = args.output if args.output else args.output_dir_pos
    
    # Handle the case when only input directory is provided (use it for both input and output)
    if input_dir and not output_dir:
        output_dir = input_dir
        print(f"No output directory specified. Using input directory '{input_dir}' for output as well.")
    
    # Ensure we have at least the input directory
    if not input_dir:
        parser.print_help()
        sys.exit(1)
    
    main(
        input_dir=input_dir,
        output_dir=output_dir,
        test_size=args.test_size,
        min_length=args.min_length,
        name=args.name
    )
