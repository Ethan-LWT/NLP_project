# -*- coding: utf-8 -*-

"""
Extracts text from WARC files and saves it to a CSV file.
This script reads WARC files, extracts the HTML content, and saves the text to a CSV file."
"""

import os
import io
import re
import gzip
import warcio
import warnings
import multiprocessing
from functools import partial
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning

# Filter out warnings
warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

def is_probably_html(content):
    """Check if content is likely to be HTML by looking for common HTML tags"""
    if not content:
        return False
    
    # Convert bytes to string if needed
    if isinstance(content, bytes):
        try:
            content_str = content.decode('utf-8', errors='ignore')
        except:
            content_str = str(content)
    else:
        content_str = content
    
    # Look for common HTML patterns
    html_patterns = [
        r'<!DOCTYPE\s+html',
        r'<html',
        r'<head',
        r'<body',
        r'<div',
        r'<p>',
        r'<a\s+href'
    ]
    
    for pattern in html_patterns:
        if re.search(pattern, content_str, re.IGNORECASE):
            return True
    
    return False

def extract_text_from_html(html_content, parser='html.parser'):
    """Extract text from HTML content using BeautifulSoup or regex depending on complexity"""
    try:
        # Use a simple tag stripper if the content is straightforward - much faster than BeautifulSoup
        if '<body' in html_content.lower():
            # Extract content between body tags with a simple regex
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
            if body_match:
                html_content = body_match.group(1)
        
        # Try faster approach first: simple regex-based tag removal
        try:
            # Remove script, style and other non-content tags first
            cleaned_content = re.sub(r'<(script|style|noscript|iframe|head|footer|nav)[^>]*>.*?</\1>', ' ', html_content, flags=re.DOTALL | re.IGNORECASE)
            # Remove all remaining HTML tags but keep their content
            text = re.sub(r'<[^>]+>', ' ', cleaned_content)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        except Exception:
            # Fallback to BeautifulSoup if regex fails
            soup = BeautifulSoup(html_content, parser)
            text = soup.get_text(separator=' ', strip=True)
    except Exception:
        # If all else fails, try the full BeautifulSoup
        try:
            soup = BeautifulSoup(html_content, parser)
            text = soup.get_text(separator=' ', strip=True)
        except Exception:
            return None
    
    # Clean up text: normalize whitespace and remove control characters
    text = re.sub(r'\s+', ' ', text).strip()
    text = ''.join(ch for ch in text if ch >= ' ' or ch in '\n\t')
    
    return text

def content_matches_topics(text, url, topics=None, domains=None):
    """
    Check if the content matches any of the specified topics or domains.
    
    Args:
        text: The extracted text content
        url: The URL of the content
        topics: List of topics to match (strings to search for in the text)
        domains: List of domains to match
        
    Returns:
        True if the content matches any topic or domain, False otherwise
    """
    if not topics and not domains:
        return True  # No filter, accept all
        
    # Check domains first (faster)
    if domains:
        for domain in domains:
            if domain.lower() in url.lower():
                return True
    
    # Check topics (more expensive)
    if topics and text:
        for topic in topics:
            if topic.lower() in text.lower():
                return True
    
    return False

def process_content(content_data):
    """
    Process pre-extracted content data (picklable) instead of WARC record objects
    Returns extracted text or None
    
    content_data is a dictionary with:
    - content: the raw content bytes
    - content_type: the content type from HTTP headers
    - url: the URL of the content
    - topics: optional list of topics to filter by
    - domains: optional list of domains to filter by
    """
    try:
        content = content_data['content']
        content_type = content_data['content_type']
        url = content_data.get('url', '')
        topics = content_data.get('topics')
        domains = content_data.get('domains')
        
        # Skip non-text content
        if not ('text/html' in content_type or 
                'text/plain' in content_type or 
                'text/xml' in content_type or 
                'application/xhtml' in content_type):
            return None
        
        # Skip if content is too small or not HTML-like
        if len(content) < 50 or not is_probably_html(content):
            return None
        
        # Choose parser based on content type - use the fastest available
        parser = 'html.parser'  # Default
        try:
            # Try to use lxml for better performance if available
            import lxml
            parser = 'lxml'
        except ImportError:
            pass
        
        # Try to detect the encoding from HTTP headers or meta tags
        detected_encoding = None
        
        # Look for charset in content-type header
        if 'charset=' in content_type:
            charset_match = re.search(r'charset=([^\s;]+)', content_type)
            if charset_match:
                detected_encoding = charset_match.group(1).strip('"\'')
        
        # Look for meta charset in HTML content
        if not detected_encoding:
            meta_match = re.search(rb'<meta[^>]+charset=(["\']?)([^"\'>]+)\1', content, re.IGNORECASE)
            if meta_match:
                detected_encoding = meta_match.group(2).decode('ascii', errors='ignore')
        
        # Try an expanded list of encodings
        encodings_to_try = [
            # First try detected encoding if any
            detected_encoding,
            # Then try common web encodings
            'utf-8', 'utf-16', 'utf-32',
            # Western European encodings
            'latin-1', 'iso-8859-1', 'iso-8859-15', 'windows-1252', 'cp1252',
            # Central/Eastern European encodings
            'iso-8859-2', 'windows-1250', 'cp1250',
            # Cyrillic encodings
            'iso-8859-5', 'windows-1251', 'cp1251', 'koi8-r', 'koi8-u',
            # Greek encoding
            'iso-8859-7', 'windows-1253',
            # Turkish encoding
            'iso-8859-9', 'windows-1254',
            # Hebrew encoding
            'iso-8859-8', 'windows-1255',
            # Arabic encoding
            'iso-8859-6', 'windows-1256',
            # Chinese encodings
            'gb2312', 'gbk', 'gb18030', 'big5', 'big5hkscs',
            # Japanese encodings
            'shift_jis', 'euc-jp', 'iso-2022-jp',
            # Korean encoding
            'euc-kr'
        ]
        
        # Filter out None and deduplicate
        encodings_to_try = [e for e in encodings_to_try if e]
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        # Try to decode content with different encodings
        decoded_content = None
        best_replacement_ratio = 1.0  # Start with worst possible ratio
        
        for encoding in encodings_to_try:
            try:
                candidate = content.decode(encoding, errors='replace')
                # Calculate the ratio of replacement characters
                replacement_ratio = candidate.count('\ufffd') / len(candidate) if len(candidate) > 0 else 1.0
                
                # If we have a perfect decode or a better ratio than before, use this result
                if replacement_ratio == 0:
                    decoded_content = candidate
                    break
                elif replacement_ratio < best_replacement_ratio:
                    best_replacement_ratio = replacement_ratio
                    decoded_content = candidate
                    
                    # If we have a good enough decode (less than 5% replacements), stop trying
                    if replacement_ratio < 0.05:
                        break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if decoded_content is None:
            # If all encodings fail, use replace mode with utf-8 as last resort
            decoded_content = content.decode('utf-8', errors='replace')
        
        # Extract the text
        text = extract_text_from_html(decoded_content, parser)
        
        # Only return text of a reasonable length
        if text and 10 < len(text.strip()) < 1000000:
            # Check if the content matches topics/domains before returning
            if content_matches_topics(text, url, topics, domains):
                return text
        
    except Exception:
        return None
    
    return None

def extract_text(warc_path, max_records=None, num_workers=None, batch_size=100, filter_topics=None, filter_domains=None):
    """
    Extract text from WARC file using parallel processing.
    Returns a list of extracted text.
    
    Args:
        warc_path: Path to the WARC file
        max_records: Maximum number of records to process (None for all)
        num_workers: Number of worker processes to use (None for auto)
        batch_size: Number of records to process in each batch
        filter_topics: List of topics to filter by
        filter_domains: List of domains to filter by
    """
    if num_workers is None:
        # Default to number of CPU cores
        num_workers = max(1, multiprocessing.cpu_count())
    
    print(f"Using {num_workers} worker processes")
    extracted_texts = []
    processed_records = 0
    error_records = 0
    
    # Instead of storing record objects, we'll extract and store content data
    # that is picklable for multiprocessing
    content_batch = []
    
    # Open the WARC file
    open_func = gzip.open if warc_path.lower().endswith('.gz') else open
    with open_func(warc_path, 'rb') as stream:
        # Create a WARCIterator
        for record in ArchiveIterator(stream):
            processed_records += 1
            
            # Skip non-response records early
            if record.rec_type != 'response':
                continue
                
            try:
                # Get HTTP headers
                http_headers = record.http_headers
                if not http_headers:
                    continue
                    
                # Get content type and URL
                content_type = http_headers.get_header('Content-Type', '').lower()
                url = record.rec_headers.get_header('WARC-Target-URI', '')
                
                # Extract content as bytes
                content = record.content_stream().read()
                
                # Create a picklable content data dictionary
                content_data = {
                    'content': content,
                    'content_type': content_type,
                    'url': url,
                    'topics': filter_topics,
                    'domains': filter_domains
                }
                
                # Add to batch
                content_batch.append(content_data)
                
                # Process batch when it reaches batch_size
                if len(content_batch) >= batch_size:
                    # Process in parallel
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        batch_results = pool.map(process_content, content_batch)
                    
                    # Filter out None results
                    valid_results = [r for r in batch_results if r is not None]
                    extracted_texts.extend(valid_results)
                    
                    error_records += len(content_batch) - len(valid_results)
                    content_batch = []
                    
                # Stop if we've reached max_records
                if max_records and processed_records >= max_records:
                    print(f"Reached maximum number of records ({max_records})")
                    break
                    
            except Exception as e:
                error_records += 1
                if error_records <= 100:
                    print(f"Error processing record {processed_records}: {type(e).__name__}: {e}")
                continue
    
    # Process any remaining records
    if content_batch:
        print(f"Processing final batch of {len(content_batch)} records")
        
        # Process in parallel
        with multiprocessing.Pool(processes=num_workers) as pool:
            batch_results = pool.map(process_content, content_batch)
        
        # Filter out None results
        valid_results = [r for r in batch_results if r is not None]
        extracted_texts.extend(valid_results)
        
        error_records += len(content_batch) - len(valid_results)
    
    print(f"Completed processing {processed_records} records")
    print(f"Successfully extracted text from {len(extracted_texts)} records")
    print(f"Encountered {error_records} errors/skipped records during processing")
    
    return extracted_texts

def process_warc(warc_path, output_dir, max_records=None, num_workers=None, batch_size=100, filter_topics=None, filter_domains=None):
    """
    Process WARC file and save extracted text to output directory.
    
    Args:
        warc_path: Path to the WARC file
        output_dir: Directory to save the output text file
        max_records: Maximum number of records to process (None for all)
        num_workers: Number of worker processes to use (None for auto)
        batch_size: Number of records to process in each batch
        filter_topics: List of topics to filter by
        filter_domains: List of domains to filter by
    """
    # Validate that warc_path is a file
    if not os.path.isfile(warc_path):
        raise ValueError(f"The WARC path '{warc_path}' is not a valid file")
    
    # Check if the file has a .warc or .warc.gz extension
    if not (warc_path.lower().endswith('.warc') or warc_path.lower().endswith('.warc.gz')):
        raise ValueError(f"The file '{warc_path}' does not have a .warc or .warc.gz extension")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the filename without extension
    base_name = os.path.basename(warc_path)
    base_name = base_name.replace('.warc.gz', '').replace('.warc', '')
    output_file = os.path.join(output_dir, base_name + '.txt')
    
    # If processing a sample or max_records, add a suffix to avoid overwriting full extractions
    if max_records:
        output_file = output_file.replace('.txt', f'_sample{max_records}.txt')
    
    print(f"Starting text extraction from {warc_path}")
    # Extract text with speed optimizations
    texts = extract_text(warc_path, max_records=max_records, 
                         num_workers=num_workers, batch_size=batch_size,
                         filter_topics=filter_topics, filter_domains=filter_domains)
    
    print(f"Writing {len(texts)} extracted texts to {output_file}")
    # Write to output file with encoding handling
    with io.open(output_file, 'w', encoding='utf-8', errors='replace') as f:
        for i, text in enumerate(texts):
            try:
                # Add a header for each document
                f.write(f"# Document {i+1}\n\n")
                f.write(text + '\n\n')
                f.write('-' * 80 + '\n\n')
            except UnicodeEncodeError:
                # Replace problematic characters
                text = text.encode('utf-8', 'replace').decode('utf-8')
                f.write(f"# Document {i+1} (with character replacements)\n\n")
                f.write(text + '\n\n')
                f.write('-' * 80 + '\n\n')
    
    return output_file
