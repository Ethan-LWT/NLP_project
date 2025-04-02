"""
Text cleaning utilities for NLP data preprocessing.
"""

import os
import re
import io
import glob
import logging
import multiprocessing
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('text_cleaning')

def detect_language(text, min_sample_length=100):
    """
    Detect the language of text using character frequency analysis.
    
    Args:
        text: Text string to analyze
        min_sample_length: Minimum text length to attempt detection
        
    Returns:
        Dictionary with detected language code and confidence score
    """
    if not text or len(text) < min_sample_length:
        return {"language": "unknown", "confidence": 0.0}
        
    # Character frequency patterns for different languages
    # These are simplified signatures of character distributions
    lang_signatures = {
        # Common Latin-based languages
        "en": {'a': 8.2, 'e': 12.7, 'i': 7.0, 'o': 7.5, 't': 9.1, 'h': 6.1, 'n': 6.8, 's': 6.3, 'r': 6.0},
        "es": {'e': 13.7, 'a': 11.5, 'o': 8.7, 's': 7.2, 'n': 7.0, 'r': 6.8, 'i': 6.3, 'd': 5.9, 'l': 5.6},
        "fr": {'e': 14.7, 'a': 7.6, 's': 7.9, 'i': 7.5, 'n': 7.1, 't': 7.2, 'r': 6.6, 'u': 6.1, 'l': 5.5},
        "de": {'e': 16.9, 'n': 10.0, 'i': 7.6, 's': 7.3, 'r': 7.0, 't': 6.1, 'a': 6.5, 'd': 5.1, 'h': 4.8},
        
        # Other language groups
        "zh": {'的': 4.0, '一': 1.2, '是': 1.0, '不': 0.9, '了': 0.8, '在': 0.8},  # Simplified Chinese
        "ja": {'の': 3.5, 'し': 1.2, 'る': 1.0, 'て': 1.9, 'に': 1.8, 'と': 1.6},  # Japanese
        "ko": {'이': 4.2, '다': 3.8, '는': 3.2, '의': 2.8, '에': 2.0},             # Korean
        "ru": {'о': 10.9, 'е': 8.5, 'а': 8.0, 'и': 7.4, 'н': 6.3, 'т': 5.7},       # Russian
        "ar": {'ا': 13.1, 'ل': 12.2, 'ي': 7.5, 'و': 6.3, 'ن': 6.1, 'م': 5.8},     # Arabic
    }
    
    # Count letter frequencies
    letter_counts = {}
    total_count = 0
    
    # Only consider letters for frequency analysis
    for char in text.lower():
        # Skip non-letter characters
        if not char.strip() or char in "0123456789.,!?;:()[]{}\"'`~@#$%^&*_+-=<>|\\/":
            continue
            
        letter_counts[char] = letter_counts.get(char, 0) + 1
        total_count += 1
    
    if total_count == 0:
        return {"language": "unknown", "confidence": 0.0}
    
    # Convert counts to percentages
    letter_freqs = {k: (v / total_count) * 100 for k, v in letter_counts.items()}
    
    # Compare with language signatures
    best_lang = "unknown"
    best_score = 0
    scores = {}
    
    for lang, signature in lang_signatures.items():
        score = 0
        matched_chars = 0
        
        # Check if key characters of language exist in text
        for char, expected_freq in signature.items():
            if char in letter_freqs:
                matched_chars += 1
                # Score based on proximity to expected frequency
                char_score = 1.0 - min(abs(letter_freqs[char] - expected_freq) / expected_freq, 1.0)
                score += char_score
        
        # Normalize score by number of signature characters
        if len(signature) > 0:
            score = score / len(signature)
            
        # Weight score by matched characters ratio
        score *= (matched_chars / len(signature)) if len(signature) > 0 else 0
        
        scores[lang] = score
        
        if score > best_score:
            best_score = score
            best_lang = lang
    
    # Check for non-Latin scripts as they're easier to detect
    # If we have Chinese/Japanese/Korean/Arabic chars, prioritize those results
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')  # CJK Unified Ideographs
    jp_chars = sum(1 for c in text if '\u3040' <= c <= '\u30ff')   # Hiragana and Katakana
    kr_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')   # Korean Hangul
    ar_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')   # Arabic
    
    if cjk_chars > 10:
        # If significant Chinese characters, prioritize Chinese over Latin languages
        if scores.get('zh', 0) > 0:
            best_lang = 'zh'
            best_score = max(best_score, scores['zh'] + 0.3)  # Boost confidence
            
    if jp_chars > 10:
        # If significant Japanese-specific characters, it's likely Japanese
        if scores.get('ja', 0) > 0:
            best_lang = 'ja'
            best_score = max(best_score, scores['ja'] + 0.3)
            
    if kr_chars > 10:
        # If significant Korean-specific characters, it's likely Korean
        if scores.get('ko', 0) > 0:
            best_lang = 'ko'
            best_score = max(best_score, scores['ko'] + 0.3)
            
    if ar_chars > 10:
        # If significant Arabic characters, prioritize Arabic
        if scores.get('ar', 0) > 0:
            best_lang = 'ar'
            best_score = max(best_score, scores['ar'] + 0.3)
    
    return {
        "language": best_lang,
        "confidence": min(best_score, 1.0),  # Cap at 1.0
        "all_scores": scores
    }

def clean_text(input_file, deep_clean=False, output_file=None):
    """
    Clean text from an input file and optionally save to an output file.
    
    Args:
        input_file: Path to the input text file
        deep_clean: Use more aggressive cleaning if True
        output_file: Path to the output file (if None, returns cleaned text)
        
    Returns:
        Cleaned text if output_file is None, otherwise None
    """
    try:
        # Read the input file
        with io.open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            
        # Apply standard cleaning
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
        cleaned_text = re.sub(r'www\.\S+', '', cleaned_text)
        
        # Remove very short lines (likely noise)
        lines = cleaned_text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 5]
        cleaned_text = '\n'.join(lines)
        
        # Apply deep cleaning if requested
        if deep_clean:
            # Remove email addresses
            cleaned_text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '', cleaned_text)
            
            # Remove special characters except punctuation
            cleaned_text = re.sub(r'[^\w\s.,!?;:\'"-]', ' ', cleaned_text)
            
            # Remove repeated punctuation
            cleaned_text = re.sub(r'([.,!?;:])\1+', r'\1', cleaned_text)
            
            # Replace multiple spaces with a single space
            cleaned_text = re.sub(r' +', ' ', cleaned_text)
            
            # Remove lines that are very likely noise (all uppercase, all numbers, etc.)
            lines = cleaned_text.split('\n')
            filtered_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Skip all uppercase lines (likely headers)
                if line.isupper() and len(line) > 10:
                    continue
                # Skip lines that are mostly numbers
                if sum(1 for c in line if c.isdigit()) > len(line) * 0.5:
                    continue
                filtered_lines.append(line)
            cleaned_text = '\n'.join(filtered_lines)
        
        # Save to output file if specified
        if output_file:
            with io.open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            return None
        else:
            return cleaned_text
            
    except Exception as e:
        logger.error(f"Error cleaning file {input_file}: {e}")
        if not output_file:
            return text  # Return original text if cleaning failed
        return None

# Define the wrapper function outside of batch_clean_files so it can be pickled
def _clean_file_wrapper(args):
    """
    Wrapper function to clean a single file.
    
    Args:
        args: Tuple of (file_path, deep_clean, input_dir, output_dir)
        
    Returns:
        True if cleaning was successful, False otherwise
    """
    file_path, deep_clean, input_dir, output_dir = args
    try:
        # Determine output path
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Make sure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Clean the file
        clean_text(file_path, deep_clean, output_path)
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def batch_clean_files(input_dir, deep_clean=False, output_dir=None, pattern='*.txt', num_workers=None):
    """
    Clean all text files in a directory.
    
    Args:
        input_dir: Directory containing text files to clean
        deep_clean: Use more aggressive cleaning if True
        output_dir: Directory to save cleaned files (if None, use input_dir)
        pattern: File pattern to match
        num_workers: Number of processes to use for parallel cleaning
        
    Returns:
        True if cleaning was successful, False otherwise
    """
    try:
        if not isinstance(input_dir, str):
            logger.error(f"Input directory path must be a string, got {type(input_dir)}")
            return False
            
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Use the same directory for output if not specified
        output_dir = output_dir or input_dir
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of files to clean
        try:
            files = glob.glob(os.path.join(input_dir, pattern))
        except Exception as e:
            logger.error(f"Error finding files with pattern '{pattern}': {e}")
            return False
        
        if not files:
            logger.warning(f"No files matching pattern '{pattern}' found in {input_dir}")
            return False
        
        logger.info(f"Found {len(files)} files to clean in {input_dir}")
        
        # Prepare arguments for each file (to avoid using a closure)
        args_list = [(file_path, deep_clean, input_dir, output_dir) for file_path in files]
        
        # Use parallel processing if multiple files
        if num_workers is None:
            # Default to max(1, available_cores - 1) workers
            num_workers = max(1, multiprocessing.cpu_count()    )
        
        # Process files in parallel or sequentially
        if num_workers > 1 and len(files) > 1:
            logger.info(f"Using {num_workers} processes for parallel cleaning")
            try:
                with multiprocessing.Pool(processes=num_workers) as pool:
                    # Use map instead of imap_unordered for better stability
                    results = pool.map(_clean_file_wrapper, args_list)
                success = sum(results)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                # Fall back to sequential processing
                logger.info("Falling back to sequential processing")
                success = sum(_clean_file_wrapper(args) for args in args_list)
        else:
            # Process files sequentially
            logger.info("Processing files sequentially")
            success = sum(_clean_file_wrapper(args) for args in args_list)
        
        logger.info(f"Successfully cleaned {success} out of {len(files)} files")
        return success > 0
        
    except Exception as e:
        logger.error(f"Error in batch cleaning: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean text files')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--output', help='Output file or directory')
    parser.add_argument('--deep', action='store_true', help='Use deep cleaning')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    parser.add_argument('--pattern', default='*.txt', help='File pattern for batch mode')
    parser.add_argument('--workers', type=int, help='Number of worker processes')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        batch_clean_files(args.input, args.deep, args.output, args.pattern, args.workers)
    else:
        clean_text(args.input, args.deep, args.output)
