"""
Common Crawl dataset generation utility for NLP projects.

This module provides tools for downloading and processing Common Crawl data
for NLP dataset creation, including:
1. Query the Common Crawl index
2. Download specific WARC files
3. Extract text content
4. Prepare training and testing datasets
"""

import os
import requests
import json
import gzip
import shutil
import time
import random
import logging
import io
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import multiprocessing
from functools import partial
from datetime import datetime
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('data_generation')

# Common Crawl constants
INDEX_URL = 'https://index.commoncrawl.org/'
CC_BASE_URL = 'https://data.commoncrawl.org/'
FALLBACK_INDEX = 'CC-MAIN-2025-08'  # Use a known working index as fallback

def get_latest_crawl_index():
    """
    Retrieve the latest Common Crawl index from the API.
    Returns the ID of the latest crawl, or a fallback value if the API call fails.
    """
    try:
        response = requests.get('https://index.commoncrawl.org/collinfo.json', 
                               headers={"User-Agent": "Mozilla/5.0"}, 
                               timeout=30)
        response.raise_for_status()
        crawls = response.json()
        
        if crawls:
            latest_index = crawls[0]["id"]
            logger.info(f"Retrieved latest Common Crawl index: {latest_index}")
            return latest_index
        else:
            logger.warning("No Common Crawl indexes found via API, using fallback")
            return FALLBACK_INDEX
    except Exception as e:
        logger.error(f"Error retrieving latest Common Crawl index: {e}")
        logger.warning(f"Falling back to default index: {FALLBACK_INDEX}")
        return FALLBACK_INDEX

# Use the latest crawl index by default
DEFAULT_INDEX = get_latest_crawl_index()

def list_available_crawls():
    """List all available Common Crawl indexes"""
    try:
        response = requests.get('https://index.commoncrawl.org/collinfo.json',
                               headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        crawls = response.json()
        
        # Return crawls from newest to oldest
        return [crawl["id"] for crawl in crawls]
    except Exception as e:
        logger.error(f"Error listing Common Crawl indexes: {e}")
        return [DEFAULT_INDEX]  # Return at least the default index

def download_warc_paths_file(crawl_id):
    """
    Download and parse the warc.paths.gz file to get a list of WARC files available.
    
    Args:
        crawl_id: Common Crawl dataset ID (e.g., 'CC-MAIN-2023-14')
        
    Returns:
        List of WARC file paths
    """
    warc_paths_url = f"{CC_BASE_URL}crawl-data/{crawl_id}/warc.paths.gz"
    
    try:
        logger.info(f"Downloading WARC paths file from: {warc_paths_url}")
        response = requests.get(warc_paths_url, stream=True, 
                               headers={"User-Agent": "Mozilla/5.0"}, 
                               timeout=60)
        response.raise_for_status()
        
        # Read and decompress the gzipped content
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
            warc_paths = gz_file.read().decode('utf-8').splitlines()
        
        # Filter for actual WARC files
        warc_paths = [path for path in warc_paths if path.endswith('.warc.gz')]
        
        logger.info(f"Found {len(warc_paths)} WARC files in paths file")
        return warc_paths
    except Exception as e:
        logger.error(f"Error downloading WARC paths file: {e}")
        return []

def filter_and_sample_warc_paths(warc_paths, max_files=5, seed=42):
    """
    Filter and randomly sample WARC paths.
    
    Args:
        warc_paths: List of WARC file paths
        max_files: Maximum number of files to select
        seed: Random seed for reproducibility
        
    Returns:
        List of selected WARC file paths
    """
    if not warc_paths:
        return []
        
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly sample if we have more files than needed
    if len(warc_paths) > max_files:
        selected_paths = random.sample(warc_paths, max_files)
        logger.info(f"Randomly selected {max_files} WARC files from {len(warc_paths)} available files")
    else:
        selected_paths = warc_paths
    
    return selected_paths

def download_warc_file(warc_path, output_dir, timeout=300):
    """
    Download a specific WARC file from Common Crawl.
    
    Args:
        warc_path: Path to the WARC file in Common Crawl storage
        output_dir: Directory to save the downloaded file
        timeout: Download timeout in seconds
        
    Returns:
        Path to the downloaded file, or None if download failed
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full URL to download
    url = f"{CC_BASE_URL}{warc_path}"
    
    # Get the filename from the path
    filename = os.path.basename(warc_path)
    output_path = os.path.join(output_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path
    
    # Create a temporary file for downloading
    temp_path = output_path + '.tmp'
    
    logger.info(f"Downloading {url} to {output_path}")
    try:
        with requests.get(url, stream=True, timeout=timeout, 
                         headers={"User-Agent": "Mozilla/5.0"}) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            # Show progress bar during download
            with open(temp_path, 'wb') as f, tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Rename the temporary file to the final filename
        shutil.move(temp_path, output_path)
        logger.info(f"Download complete: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        # Clean up the temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def download_multiple_warcs(warc_paths, output_dir, max_workers=4):
    """
    Download multiple WARC files in parallel.
    
    Args:
        warc_paths: List of WARC file paths to download
        output_dir: Directory to save the downloaded files
        max_workers: Maximum number of parallel downloads
        
    Returns:
        List of successfully downloaded file paths
    """
    # Use a smaller number if there are fewer files
    workers = min(max_workers, len(warc_paths))
    
    if workers <= 1 or len(warc_paths) <= 1:
        # Sequential download for small numbers of files
        return [download_warc_file(path, output_dir) for path in warc_paths if path]
    else:
        # Parallel download for multiple files
        with multiprocessing.Pool(workers) as pool:
            download_func = partial(download_warc_file, output_dir=output_dir)
            results = list(tqdm(
                pool.imap(download_func, warc_paths),
                total=len(warc_paths),
                desc="Downloading WARC files"
            ))
        
        # Filter out failed downloads (None results)
        return [path for path in results if path]

def extract_warc_records(warc_paths, output_dir, num_workers=None, max_records_per_file=None, batch_size=100, topics=None, domains=None):
    """
    Extract text from multiple WARC files.
    
    Args:
        warc_paths: List of paths to WARC files
        output_dir: Directory to save the extracted records
        num_workers: Number of worker processes to use for extraction
        max_records_per_file: Maximum records to extract from each file
        batch_size: Number of records to process in each batch
        topics: Optional list of topics to filter content by
        domains: Optional list of domains to filter content by
        
    Returns:
        List of paths to the extracted files
    """
    # Import load_textdata here to avoid circular imports
    from load_textdata import process_warc
    
    extracted_files = []
    
    # Process each WARC file
    for warc_path in warc_paths:
        if not warc_path or not os.path.exists(warc_path):
            logger.warning(f"Skipping invalid or missing WARC file: {warc_path}")
            continue
            
        try:
            logger.info(f"Extracting text from {warc_path}")
            output_file = process_warc(
                warc_path, 
                output_dir,
                max_records=max_records_per_file,
                num_workers=num_workers,
                batch_size=batch_size,
                filter_topics=topics,      # Pass topics for content filtering
                filter_domains=domains     # Pass domains for content filtering
            )
            extracted_files.append(output_file)
            logger.info(f"Extraction complete: {output_file}")
        except Exception as e:
            logger.error(f"Error processing WARC file {warc_path}: {e}")
    
    return extracted_files

def create_dataset_from_common_crawl(
    output_dir,
    topics=None,
    domains=None,
    limit_per_query=50,
    crawl_id=DEFAULT_INDEX,
    max_warc_downloads=5,
    max_records_per_warc=100,
    split_ratio=0.8,
    clean=True,
    deep_clean=False,
    seed=42,
    max_workers=None,
    batch_size=100
):
    """
    Create an NLP dataset from Common Crawl data.
    
    Args:
        output_dir: Directory to save the created dataset
        topics: List of topics to filter content by
        domains: List of domains to filter content by
        limit_per_query: Maximum number of results per query (for index approach)
        crawl_id: Common Crawl dataset ID to use
        max_warc_downloads: Maximum number of WARC files to download
        max_records_per_warc: Maximum records to extract from each WARC file
        split_ratio: Train/test split ratio (0.0-1.0)
        clean: Whether to clean the extracted text
        deep_clean: Whether to use deep cleaning
        seed: Random seed for reproducibility
        max_workers: Maximum number of worker processes for parallel operations
        batch_size: Number of records to process in each batch
        
    Returns:
        Dictionary with paths to created training and testing files
    """
    from text_cleaning import batch_clean_files
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    download_dir = os.path.join(output_dir, 'downloads')
    os.makedirs(download_dir, exist_ok=True)
    extracted_dir = os.path.join(output_dir, 'extracted')
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Use default topics if none provided
    if not topics and not domains:
        topics = ["artificial intelligence", "machine learning", "natural language processing"]
        domains = ["wikipedia.org", "archive.org"]
        logger.info("No topics or domains provided, using defaults")
    
    # Log the topics and domains we'll use for filtering
    if topics:
        logger.info(f"Using topics for filtering: {topics}")
    if domains:
        logger.info(f"Using domains for filtering: {domains}")
        
    # Use direct warc.paths.gz approach to get WARC files
    logger.info(f"Getting WARC file paths from crawl dataset: {crawl_id}")
    warc_paths = download_warc_paths_file(crawl_id)
    
    if not warc_paths:
        logger.error(f"No WARC files found in crawl dataset: {crawl_id}")
        return None
    
    # Filter and sample the WARC paths
    selected_warc_paths = filter_and_sample_warc_paths(warc_paths, max_warc_downloads, seed)
    
    if not selected_warc_paths:
        logger.error("No WARC files were selected for download")
        return None
    
    # Download the WARC files
    logger.info(f"Downloading {len(selected_warc_paths)} WARC files")
    downloaded_paths = download_multiple_warcs(selected_warc_paths, download_dir, max_workers=max_workers or 4)
    
    if not downloaded_paths:
        logger.error("No WARC files were successfully downloaded")
        return None
    
    logger.info(f"Successfully downloaded {len(downloaded_paths)} WARC files")
    
    # Extract text from the WARC files
    extracted_files = extract_warc_records(
        downloaded_paths,
        extracted_dir,
        num_workers=max_workers,
        max_records_per_file=max_records_per_warc,
        batch_size=batch_size,
        topics=topics,
        domains=domains
    )
    
    if not extracted_files:
        logger.error("No text was successfully extracted from WARC files")
        return None
    
    logger.info(f"Successfully extracted text to {len(extracted_files)} files")
    
    # Clean the text files if requested
    if clean:
        logger.info(f"Cleaning extracted text files using deep_clean={deep_clean}")
        batch_clean_files(extracted_dir, deep_clean, extracted_dir)
    
    # Create train and test datasets
    train_file = os.path.join(output_dir, 'train.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    # Combine all extracted files into a single collection
    all_texts = []
    for file_path in extracted_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                if content:
                    # Split the content by the document separator to get individual documents
                    docs = content.split('-' * 80)
                    for doc in docs:
                        doc = doc.strip()
                        if doc and len(doc) > 100:  # Skip very short documents
                            all_texts.append(doc)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    if not all_texts:
        logger.error("No valid text content found in extracted files")
        return None
    
    logger.info(f"Collected {len(all_texts)} documents from extracted files")
    
    # Shuffle and split the data
    random.seed(seed)
    random.shuffle(all_texts)
    
    split_index = int(len(all_texts) * split_ratio)
    train_texts = all_texts[:split_index]
    test_texts = all_texts[split_index:]
    
    if not train_texts or not test_texts:
        logger.error("Not enough text content to create both training and testing sets")
        return None
    
    # Write train and test files
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_texts))
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_texts))
    
    logger.info(f"Created training dataset: {train_file} ({len(train_texts)} documents)")
    logger.info(f"Created testing dataset: {test_file} ({len(test_texts)} documents)")
    
    # Create a metadata file with details about the dataset
    metadata_file = os.path.join(output_dir, 'dataset_metadata.json')
    metadata = {
        'created_at': datetime.now().isoformat(),
        'crawl_id': crawl_id,
        'topics': topics,
        'domains': domains,
        'warc_files_downloaded': len(downloaded_paths),
        'extracted_files': len(extracted_files),
        'total_documents': len(all_texts),
        'train_documents': len(train_texts),
        'test_documents': len(test_texts),
        'split_ratio': split_ratio,
        'cleaned': clean,
        'deep_cleaned': deep_clean,
        'seed': seed
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset metadata saved to {metadata_file}")
    
    return {
        'train_file': train_file,
        'test_file': test_file,
        'extracted_files': extracted_files,
        'downloaded_files': downloaded_paths,
        'metadata_file': metadata_file
    }

def run_dataset_generation(
    output_dir,
    topics=None,
    domains=None,
    limit_per_query=50,
    crawl_index=None,
    max_warcs=5,
    max_records=100,
    split_ratio=0.8,
    clean=True,
    deep_clean=False,
    seed=42,
    max_workers=None,
    batch_size=100,
    force_latest=False
):
    """
    Command line interface function for Common Crawl dataset creation
    
    Args:
        output_dir: Directory to save the created dataset
        topics: List of topics to filter content by
        domains: List of domains to filter content by
        limit_per_query: Maximum number of results per query
        crawl_index: Common Crawl dataset ID to use (None = latest)
        max_warcs: Maximum number of WARC files to download
        max_records: Maximum records per WARC file
        split_ratio: Train/test split ratio (0.0-1.0)
        clean: Whether to clean the extracted text
        deep_clean: Whether to use deep cleaning
        seed: Random seed
        max_workers: Maximum number of worker processes
        batch_size: Number of records per batch
        force_latest: If True, always try to get latest index even if crawl_index is specified
    """
    # Determine which crawl index to use
    if force_latest or not crawl_index:
        available_crawls = list_available_crawls()
        if available_crawls:
            crawl_id = available_crawls[0]
            logger.info(f"Using latest available crawl index: {crawl_id}")
        else:
            crawl_id = DEFAULT_INDEX
            logger.warning(f"Unable to determine latest crawl index. Using default: {DEFAULT_INDEX}")
    else:
        crawl_id = crawl_index
        logger.info(f"Using specified crawl index: {crawl_id}")
            
    return create_dataset_from_common_crawl(
        output_dir=output_dir,
        topics=topics,
        domains=domains,
        limit_per_query=limit_per_query,
        crawl_id=crawl_id,
        max_warc_downloads=max_warcs,
        max_records_per_warc=max_records,
        split_ratio=split_ratio,
        clean=clean,
        deep_clean=deep_clean,
        seed=seed,
        max_workers=max_workers,
        batch_size=batch_size
    )

def main():
    """Command line interface for dataset generation"""
    parser = argparse.ArgumentParser(description='Create NLP datasets from Common Crawl')
    
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save the created dataset')
    parser.add_argument('--topics', nargs='+',
                       help='List of topics to filter content by (e.g., "machine learning" "python")')
    parser.add_argument('--domains', nargs='+',
                       help='List of domains to filter content by (e.g., "wikipedia.org" "example.com")')
    parser.add_argument('--limit', type=int, default=50,
                       help='Maximum results per query (not used in direct download mode)')
    parser.add_argument('--crawl_index', default=None,
                       help='Common Crawl dataset ID to use (defaults to latest available)')
    parser.add_argument('--force_latest', action='store_true',
                       help='Force using the latest crawl dataset even if another is specified')
    parser.add_argument('--max_warcs', type=int, default=5,
                       help='Maximum number of WARC files to download')
    parser.add_argument('--max_records', type=int, default=100,
                       help='Maximum records to extract from each WARC file')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/test split ratio (0.0-1.0)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean the extracted text')
    parser.add_argument('--deep_clean', action='store_true',
                       help='Use deep cleaning')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--list_crawls', action='store_true',
                       help='List available Common Crawl datasets')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of worker processes to use (default: auto)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Number of records to process in each batch')
    
    args = parser.parse_args()
    
    if args.list_crawls:
        crawls = list_available_crawls()
        print("\nAvailable Common Crawl indexes (newest first):")
        for i, crawl in enumerate(crawls):
            if i == 0:
                print(f"  {crawl} (latest)")
            else:
                print(f"  {crawl}")
        
        # Print a note about the default
        print(f"\nThe default/latest index is: {DEFAULT_INDEX}")
        print("This index will be used if no specific index is requested.")
        return
    
    logger.info(f"Starting dataset generation from Common Crawl")
    if args.crawl_index:
        logger.info(f"Using specified crawl index: {args.crawl_index}")
        if args.force_latest:
            logger.info("Note: --force_latest will override the specified crawl index with the latest available")
    else:
        logger.info(f"Using latest available crawl index")
        
    result = run_dataset_generation(
        output_dir=args.output_dir,
        topics=args.topics,
        domains=args.domains,
        limit_per_query=args.limit,
        crawl_index=args.crawl_index,
        max_warcs=args.max_warcs,
        max_records=args.max_records,
        split_ratio=args.split_ratio,
        clean=args.clean,
        deep_clean=args.deep_clean,
        seed=args.seed,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        force_latest=args.force_latest
    )
    
    if result:
        print("\nDataset creation complete!")
        print(f"Training file: {result['train_file']}")
        print(f"Testing file: {result['test_file']}")
        print(f"Downloaded {len(result['downloaded_files'])} WARC files")
        print(f"Created {len(result['extracted_files'])} extracted files")
        print(f"Metadata file: {result['metadata_file']}")
    else:
        print("Dataset creation failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
