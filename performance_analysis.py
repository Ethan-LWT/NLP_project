"""
Performance analysis for text cleaning operations.

This module provides tools to measure and compare the performance of
different text cleaning methods, particularly focusing on the performance
differences between regular and deep cleaning.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from text_cleaning import clean_text, detect_language
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
import io

# Constants for text analysis
HTML_PATTERN = re.compile(r'<[^>]+>')
NOISE_PATTERN = re.compile(r'[^\w\s.,;:!?\'"-]')
SENTENCE_PATTERN = re.compile(r'[.!?]+')

def analyze_text_sample(text, full_analysis=False):
    """
    Analyze a sample of text to determine quality metrics
    This is a faster version that works on a sample rather than the full text
    
    Args:
        text (str): Text to analyze
        full_analysis (bool): Whether to perform deep analysis (slower)
        
    Returns:
        dict: Quality metrics
    """
    sample_size = min(len(text), 10000)
    if len(text) > sample_size:
        begin = text[:sample_size//3]
        middle_start = len(text)//2 - sample_size//6
        middle = text[middle_start:middle_start + sample_size//3]
        end = text[-sample_size//3:]
        sample = begin + middle + end
    else:
        sample = text
    
    metrics = {
        'text_length': len(text),
        'sample_length': len(sample)
    }
    
    metrics['has_html'] = '<' in sample and '>' in sample
    
    if metrics['has_html']:
        html_count = sample.count('<')
        verified_count = len(HTML_PATTERN.findall(sample[:1000]))
        if verified_count > 0:
            ratio = verified_count / sample[:1000].count('<')
            html_count = int(html_count * ratio)
        metrics['html_ratio'] = html_count / len(sample) if len(sample) > 0 else 0
    else:
        metrics['html_ratio'] = 0
    
    noise_chars = sum(1 for char in sample[:2000] if not char.isalnum() and not char.isspace())
    metrics['noise_ratio'] = noise_chars / min(2000, len(sample)) if len(sample) > 0 else 0
    
    sentence_count = sum(1 for char in sample if char in '.!?')
    metrics['sentence_count'] = max(1, sentence_count)
    
    word_count = len(sample.split())
    metrics['avg_sentence_length'] = word_count / metrics['sentence_count']
    
    if full_analysis:
        clean_sample = sample[:2000]
        
        regular_cleaned = clean_text(clean_sample, deep=False)
        metrics['regular_reduction'] = (1 - len(regular_cleaned) / len(clean_sample)) * 100 if len(clean_sample) > 0 else 0
        
        deep_cleaned = clean_text(clean_sample, deep=True)
        metrics['deep_reduction'] = (1 - len(deep_cleaned) / len(clean_sample)) * 100 if len(clean_sample) > 0 else 0
        
        metrics['additional_reduction'] = metrics['deep_reduction'] - metrics['regular_reduction']
    else:
        metrics['regular_reduction'] = min(80, metrics['html_ratio'] * 100 + metrics['noise_ratio'] * 50)
        metrics['deep_reduction'] = min(90, metrics['regular_reduction'] * 1.3)
        metrics['additional_reduction'] = metrics['deep_reduction'] - metrics['regular_reduction']
    
    return metrics

def read_file_chunk(file_path, start_pos, size):
    """Read a chunk of a file from a specific position"""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        f.seek(start_pos)
        if start_pos > 0:
            f.readline()
        return f.read(size)

def analyze_file_fast(file_path, sample_size=None):
    """
    Fast analysis of a file by reading only portions of it
    
    Args:
        file_path (str): Path to the text file
        sample_size (int, optional): Size of each sample to read
        
    Returns:
        dict: Performance statistics
    """
    try:
        file_size = os.path.getsize(file_path)
        
        if file_size < 500_000:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read(100_000)
            return analyze_text_sample(text, full_analysis=True)
        
        sample_size = sample_size or min(50_000, file_size // 20)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(read_file_chunk, file_path, 0, sample_size),
                executor.submit(read_file_chunk, file_path, file_size // 2 - sample_size // 2, sample_size),
                executor.submit(read_file_chunk, file_path, max(0, file_size - sample_size), sample_size)
            ]
            samples = [future.result() for future in futures]
        
        combined_sample = ''.join(samples)
        
        results = analyze_text_sample(combined_sample, full_analysis=True)
        
        results['text_length'] = file_size
        
        return results
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def analyze_file(file_path):
    """Analyze a text file and return quality metrics"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None
        
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"Analyzing file: {file_path} ({file_size:.1f} MB)")
        
        # Initialize metrics
        metrics = {
            'file_size_mb': file_size,
            'line_count': 0,
            'avg_line_length': 0,
            'avg_word_count': 0,
            'vocabulary_size': 0,
            'special_char_ratio': 0,
            'duplicated_line_ratio': 0,
            'empty_line_ratio': 0,
            'malformed_line_ratio': 0
        }
        
        # Sample lines if file is too large
        max_lines_to_read = 10000  # Maximum number of lines to read for analysis
        
        # Read the file safely, handling potential encoding issues or escape character problems
        lines = []
        seen_lines = set()
        empty_lines = 0
        malformed_lines = 0
        total_length = 0
        total_words = 0
        all_words = []
        special_chars = 0
        total_chars = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if i >= max_lines_to_read and file_size > 10:  # Sample for large files
                        break
                    
                    # Clean line safely
                    safe_line = line.strip()
                    
                    # Count empty lines
                    if not safe_line:
                        empty_lines += 1
                        continue
                    
                    # Check for malformed lines (e.g., very short or contains unusual patterns)
                    if len(safe_line) < 3 or (safe_line.count('[') > safe_line.count(']')):
                        malformed_lines += 1
                    
                    # Process line statistics
                    lines.append(safe_line)
                    total_length += len(safe_line)
                    
                    # Count words (safely)
                    try:
                        words = safe_line.split()
                        total_words += len(words)
                        all_words.extend(words)
                    except Exception:
                        # If splitting fails, count conservatively
                        word_estimate = max(1, len(safe_line) // 5)  # Rough estimate
                        total_words += word_estimate
                    
                    # Count unique lines
                    seen_lines.add(safe_line)
                    
                    # Count special characters
                    for char in safe_line:
                        total_chars += 1
                        if not char.isalnum() and not char.isspace():
                            special_chars += 1
        except UnicodeDecodeError:
            print("Warning: Unicode decoding error. Trying with latin-1 encoding...")
            try:
                with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                    # Simplified reading for latin-1 encoding
                    lines = [line.strip() for line in f.readlines()[:max_lines_to_read]]
                    empty_lines = sum(1 for line in lines if not line)
                    total_length = sum(len(line) for line in lines if line)
                    total_words = sum(len(line.split()) for line in lines if line)
                    all_words = [word for line in lines if line for word in line.split()]
                    seen_lines = set(lines)
                    special_chars = sum(1 for line in lines for char in line 
                                      if not char.isalnum() and not char.isspace())
                    total_chars = sum(len(line) for line in lines)
            except Exception as e:
                print(f"Error reading file with latin-1 encoding: {e}")
                return None
        except Exception as e:
            print(f"Error analyzing file: {e}")
            # Try to continue with partial data if possible
            if not lines:
                return None
        
        # Calculate metrics
        line_count = len(lines)
        if line_count == 0:
            print("Error: No valid lines found in file")
            return None
        
        metrics['line_count'] = line_count
        metrics['avg_line_length'] = total_length / line_count if line_count > 0 else 0
        metrics['avg_word_count'] = total_words / line_count if line_count > 0 else 0
        metrics['vocabulary_size'] = len(set(all_words))
        metrics['special_char_ratio'] = special_chars / total_chars if total_chars > 0 else 0
        metrics['duplicated_line_ratio'] = 1 - (len(seen_lines) / line_count) if line_count > 0 else 0
        metrics['empty_line_ratio'] = empty_lines / (line_count + empty_lines) if (line_count + empty_lines) > 0 else 0
        metrics['malformed_line_ratio'] = malformed_lines / line_count if line_count > 0 else 0
        
        return metrics
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def analyze_file(file_path, output_dir=None, fast_mode=True):
    """
    Analyze performance for a single file, with fast mode option
    
    Args:
        file_path (str): Path to the text file
        output_dir (str, optional): Directory to save charts
        fast_mode (bool): Whether to use fast analysis
        
    Returns:
        dict: Performance statistics
    """
    start_time = time.time()
    
    try:
        file_size = os.path.getsize(file_path)
        print(f"Analyzing file: {file_path} ({file_size/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"Error getting file size: {e}")
        return None
    
    results = analyze_file_fast(file_path)
    
    if results:
        analysis_time = time.time() - start_time
        print(f"Analysis completed in {analysis_time:.2f} seconds")
        
        print("\nQuality Analysis Results:")
        print(f"Text length: {results['text_length']} characters")
        print(f"HTML content ratio: {results['html_ratio']:.4f}")
        print(f"Noise ratio: {results['noise_ratio']:.4f}")
        print(f"Average sentence length: {results.get('avg_sentence_length', 0):.1f} words")
        print(f"Regular cleaning reduction: {results['regular_reduction']:.2f}%")
        print(f"Deep cleaning reduction: {results['deep_reduction']:.2f}%")
        print(f"Additional reduction from deep cleaning: {results['additional_reduction']:.2f}%")
        
        if output_dir:
            generate_report(results, output_dir)
    
    return results

def compute_quality_score(results):
    """
    Compute an overall quality score based on analysis results.
    Returns a score between 0.0 (poor) and 1.0 (excellent).
    """
    if not results:
        return 0.0
    
    metrics = {}
    metrics['text_length'] = min(1.0, results.get('text_length', 0) / 10000)
    metrics['regular_reduction'] = min(1.0, results.get('regular_reduction', 0) / 100)
    metrics['deep_reduction'] = min(1.0, results.get('deep_reduction', 0) / 100)
    
    avg_sentence_length = results.get('avg_sentence_length', 0)
    metrics['sentence_length'] = min(1.0, avg_sentence_length / 20) if avg_sentence_length < 50 else (1.0 - min(1.0, (avg_sentence_length - 50) / 100))
    
    html_ratio = results.get('html_ratio', 0)
    metrics['html_cleanliness'] = 1.0 - min(1.0, html_ratio)
    
    noise_ratio = results.get('noise_ratio', 0)
    metrics['noise_level'] = 1.0 - min(1.0, noise_ratio * 10)
    
    weights = {
        'text_length': 0.15,
        'regular_reduction': 0.1,
        'deep_reduction': 0.1,
        'sentence_length': 0.2,
        'html_cleanliness': 0.25,
        'noise_level': 0.2
    }
    
    score = 0.0
    total_weight = 0.0
    
    for metric, value in metrics.items():
        if metric in weights:
            score += value * weights[metric]
            total_weight += weights[metric]
    
    if total_weight > 0:
        return score / total_weight
    return 0.5

def clean_text_from_analysis(input_file, output_file, analysis_results):
    """
    Clean text based on analysis results.
    Uses targeted cleaning approaches based on identified issues.
    """
    from text_cleaning import clean_text
    
    use_deep_clean = True
    
    if analysis_results:
        html_ratio = analysis_results.get('html_ratio', 0)
        noise_ratio = analysis_results.get('noise_ratio', 0)
        regular_reduction = analysis_results.get('regular_reduction', 0)
        deep_reduction = analysis_results.get('deep_reduction', 0)
        
        if deep_reduction - regular_reduction < 10 and regular_reduction > 30:
            use_deep_clean = False
            print("Using regular cleaning (sufficient for this dataset)")
    
    clean_text(input_file, use_deep_clean, output_file)
    
    return use_deep_clean

def generate_report(results, output_dir):
    """Generate visual reports from analysis results"""
    try:
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        labels = ['Original', 'Regular Cleaning', 'Deep Cleaning']
        sizes = [
            results.get('text_length', 0),
            results.get('text_length', 0) * (1 - results.get('regular_reduction', 0)/100),
            results.get('text_length', 0) * (1 - results.get('deep_reduction', 0)/100)
        ]
        plt.bar(labels, sizes, color=['gray', 'blue', 'red'])
        plt.title('Text Size Comparison')
        plt.ylabel('Characters')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(sizes):
            if i == 0:
                plt.text(i, v + 0.05, f"{int(v)} chars", ha='center')
            else:
                reduction = 100 * (1 - v / sizes[0])
                plt.text(i, v + 0.05, f"{int(v)} chars\n(-{reduction:.1f}%)", ha='center')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'text_reduction.png')
        plt.savefig(chart_path)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        quality_score = compute_quality_score(results)
        quality_metrics = {
            'Overall Score': quality_score,
            'HTML Cleanliness': 1.0 - results.get('html_ratio', 0),
            'Text Quality': min(1.0, results.get('avg_sentence_length', 0) / 20) 
                           if results.get('avg_sentence_length', 0) < 50 
                           else (1.0 - min(1.0, (results.get('avg_sentence_length', 0) - 50) / 100)),
            'Noise Level': 1.0 - min(1.0, results.get('noise_ratio', 0) * 10)
        }
        
        plt.bar(quality_metrics.keys(), quality_metrics.values(), color='green')
        plt.title('Quality Metrics (higher is better)')
        plt.ylabel('Score (0-1)')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, (k, v) in enumerate(quality_metrics.items()):
            plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        metrics_path = os.path.join(output_dir, 'quality_metrics.png')
        plt.savefig(metrics_path)
        plt.close()
        
        return [chart_path, metrics_path]
    
    except Exception as e:
        print(f"Error generating report: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze text cleaning performance.')
    parser.add_argument('--file', '-f', required=True, help='Path to the text file to analyze')
    parser.add_argument('--output', '-o', help='Output directory for charts (optional)')
    parser.add_argument('--fast', '-s', action='store_true', default=True, 
                      help='Use fast analysis mode (default: True)')
    parser.add_argument('--full', '-d', action='store_false', dest='fast',
                      help='Use detailed analysis mode (slower but more accurate)')
    
    args = parser.parse_args()
    
    analyze_file(args.file, args.output, args.fast)

if __name__ == "__main__":
    main()
