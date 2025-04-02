import argparse
import os
import sys
import glob
import re
import time
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='NLP Neural Network Pipeline')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'predict', 'visualize', 'create_dataset', 'analyze'],
                       help='Mode to run: train, predict, visualize, create_dataset, or analyze')
    
    # Training arguments
    parser.add_argument('--model_type', type=str, default='classifier', 
                       choices=['classifier', 'seq2seq'],
                       help='Type of model to train (classifier or seq2seq)')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    
    # Data and output paths - original options
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to preprocessed data file with both training and testing data')
    parser.add_argument('--dataset_path', type=str, default='download_and_create_textdata',
                       help='Path to directory containing dataset_builder.py')
    
    # New options for direct specification of training and testing data
    parser.add_argument('--training_data', type=str, default=None,
                       help='Path to training data file')
    parser.add_argument('--testing_data', type=str, default=None,
                       help='Path to testing data file')
    parser.add_argument('--training_labels', type=str, default=None,
                       help='Path to training labels file (if separate from training data)')
    parser.add_argument('--testing_labels', type=str, default=None,
                       help='Path to testing labels file (if separate from testing data)')
    
    # Output paths
    parser.add_argument('--output_path', type=str, default='models',
                       help='Path to save/load models')
    parser.add_argument('--log_path', type=str, default='logs',
                       help='Path to save logs')
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save the model (overrides output_path for model saving)')
    
    # Prediction arguments
    parser.add_argument('--text', type=str, 
                       help='Text to classify in predict mode or visualize in visualize mode')
    parser.add_argument('--model_file', type=str, default='final_model.h5',
                       help='Model filename to use for prediction or visualization')
    parser.add_argument('--vectorizer_file', type=str, default='vectorizer.pkl',
                       help='Vectorizer filename to use for prediction or visualization')
    
    # Visualization arguments
    parser.add_argument('--vis_output', type=str, default='neural_network_visualization',
                       help='Output filename for visualization video (without extension)')
    parser.add_argument('--vis_quality', type=str, default='medium_quality',
                       choices=['low_quality', 'medium_quality', 'high_quality'],
                       help='Rendering quality for visualization')
    parser.add_argument('--vis_words', type=str, 
                       default="natural,language,processing,neural,network",
                       help='Comma-separated words for embedding visualization')
    
    # Add GPU control options
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU acceleration if available')
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu',
                       help='Disable GPU acceleration')
    
    # Performance optimization arguments
    parser.add_argument('--optimizer', type=str, default='lion', 
                       choices=['lion', 'adam', 'adamw', 'sgd'], 
                       help='Optimization algorithm to use')
    parser.add_argument('--use_amp', action='store_true', 
                       help='Enable automatic mixed-precision training')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable automatic mixed-precision training')
    
    # Dataset creation arguments (merged from textdata_creation.py)
    parser.add_argument('--input_dir', type=str,
                       help='Directory containing WARC files to process for dataset creation')
    parser.add_argument('--output_dir', type=str, default='textdata',
                       help='Directory to save extracted text for dataset creation')
    parser.add_argument('--max_records', type=int, default=None,
                       help='Maximum number of records to process per file')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--file_batch_size', type=int, default=100,
                       help='Number of records to process in each batch')
    parser.add_argument('--sample', type=int, default=None,
                       help='Extract a random sample of N records (for testing)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Recursively process WARC files in subdirectories')
    parser.add_argument('--file_pattern', default='*.warc',
                       help='File pattern to match WARC files (default: *.warc)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean the extracted text files')
    parser.add_argument('--clean_only', action='store_true',
                       help='Only clean existing text files, skip extraction')
    parser.add_argument('--deep_clean', action='store_true',
                       help='Perform more aggressive text cleaning')
    parser.add_argument('--force_process', action='store_true',
                       help='Process files even if output already exists')
    
    # Common Crawl dataset generation arguments
    parser.add_argument('--common_crawl', action='store_true',
                       help='Generate dataset from Common Crawl instead of local files')
    parser.add_argument('--topics', nargs='+',
                       help='List of topics to query in Common Crawl (e.g., "machine learning" "python")')
    parser.add_argument('--domains', nargs='+',
                       help='List of domains to query in Common Crawl (e.g., "wikipedia.org" "example.com")')
    parser.add_argument('--limit_per_query', type=int, default=50,
                       help='Maximum results per query for Common Crawl')
    parser.add_argument('--crawl_index', default=None,
                       help='Common Crawl index to use (defaults to latest available)')
    parser.add_argument('--max_warcs', type=int, default=5,
                       help='Maximum number of WARC files to download from Common Crawl')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/test split ratio for Common Crawl dataset (0.0-1.0)')
    parser.add_argument('--list_crawls', action='store_true',
                       help='List available Common Crawl indexes and exit')
    
    # Dataset quality analysis options
    parser.add_argument('--analyze_quality', action='store_true',
                       help='Analyze dataset quality before training')
    parser.add_argument('--auto_clean', action='store_true',
                       help='Automatically clean data if quality is poor')
    parser.add_argument('--quality_threshold', type=float, default=0.7,
                       help='Quality threshold for automatic cleaning (0.0-1.0)')
    parser.add_argument('--skip_quality_check', action='store_true',
                       help='Skip dataset quality analysis and proceed directly to training')
    
    # Common Crawl integration with training
    parser.add_argument('--create_and_train', action='store_true',
                       help='Create dataset from Common Crawl and immediately train a model')
    parser.add_argument('--force_latest', action='store_true',
                       help='Force using the latest Common Crawl index even if another is specified')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    if args.data_path:
        data_path = os.path.abspath(args.data_path)
    else:
        data_path = None
        
    dataset_path = os.path.abspath(args.dataset_path) if args.dataset_path else None
    output_path = os.path.abspath(args.save_model if args.save_model else args.output_path)
    log_path = os.path.abspath(args.log_path) if args.log_path else 'logs'
    
    # Process the training and testing data paths
    training_data = os.path.abspath(args.training_data) if args.training_data else None
    testing_data = os.path.abspath(args.testing_data) if args.testing_data else None
    training_labels = os.path.abspath(args.training_labels) if args.training_labels else None
    testing_labels = os.path.abspath(args.testing_labels) if args.testing_labels else None
    
    # Add current directory to path to find modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Handle combined dataset creation and training workflow
    if args.create_and_train:
        if args.mode != 'train':
            print("Error: --create_and_train requires --mode train")
            sys.exit(1)
            
        print("Starting combined dataset creation and model training workflow...")
        start_time = time.time()
        
        # Temporary output directory for dataset
        dataset_output_dir = os.path.join(output_path, 'generated_dataset')
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Step 1: Generate dataset from Common Crawl
        print("\nStep 1: Generating dataset from Common Crawl...")
        from data_generation import run_dataset_generation
        
        dataset_result = run_dataset_generation(
            output_dir=dataset_output_dir,
            topics=args.topics,
            domains=args.domains,
            limit_per_query=args.limit_per_query if hasattr(args, 'limit_per_query') else 50,
            crawl_index=args.crawl_index,
            max_warcs=args.max_warcs if hasattr(args, 'max_warcs') else 5,
            max_records=args.max_records,
            split_ratio=args.split_ratio if hasattr(args, 'split_ratio') else 0.8,
            clean=True,  # Always clean data in combined workflow
            deep_clean=args.deep_clean,
            seed=42,
            max_workers=args.num_workers,
            batch_size=args.file_batch_size if hasattr(args, 'file_batch_size') else 100,
            force_latest=args.force_latest
        )
        
        if not dataset_result:
            print("Error: Dataset creation failed. Cannot proceed to training.")
            sys.exit(1)
            
        print("\nDataset creation complete!")
        print(f"Training file: {dataset_result['train_file']}")
        print(f"Testing file: {dataset_result['test_file']}")
        
        # Update training and testing paths to use the newly created dataset
        training_data = dataset_result['train_file']
        testing_data = dataset_result['test_file']
        
        # Step 2: Proceed with the training process including quality analysis
        print("\nStep 2: Analyzing dataset quality and training model...")
        
        from train_model import train_model
        from performance_analysis import analyze_file, compute_quality_score, clean_text_from_analysis
        
        # Quality analysis if not skipped
        should_preprocess = False
        if not args.skip_quality_check and (args.analyze_quality or args.auto_clean):
            print(f"Analyzing training data quality: {training_data}")
            results = analyze_file(training_data)
            
            if results:
                print("\nTraining Data Quality Analysis:")
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
                # Calculate overall quality score (0.0-1.0)
                quality_score = compute_quality_score(results)
                print(f"\nOverall quality score: {quality_score:.2f}/1.0")
                
                # Decide if preprocessing is needed based on quality score
                if quality_score < args.quality_threshold:
                    print(f"Dataset quality below threshold ({args.quality_threshold})")
                    if args.auto_clean:
                        should_preprocess = True
                        print("Auto-cleaning will be performed")
                    else:
                        print("Consider using --auto_clean to automatically clean the dataset")
                else:
                    print(f"Dataset quality is good (above threshold of {args.quality_threshold})")
            else:
                print("Warning: Quality analysis failed. Proceeding without analysis.")
        
        # Preprocess data if needed and preprocessing is enabled
        if should_preprocess:
            print("\nPreprocessing data before training...")
            
            # Create a preprocessed version of the training data
            preprocessed_dir = os.path.join(output_path, 'preprocessed')
            os.makedirs(preprocessed_dir, exist_ok=True)
            
            preprocessed_training = os.path.join(preprocessed_dir, 
                                               os.path.basename(training_data))
            
            print(f"Cleaning training data: {training_data} -> {preprocessed_training}")
            start_clean_time = time.time()
            
            # Use the results from analysis to guide cleaning
            clean_text_from_analysis(training_data, preprocessed_training, results)
            
            clean_duration = time.time() - start_clean_time
            
            # If testing data exists, also preprocess it
            if testing_data:
                # Analyze testing data
                test_results = analyze_file(testing_data)
                preprocessed_testing = os.path.join(preprocessed_dir, 
                                                  os.path.basename(testing_data))
                print(f"Cleaning testing data: {testing_data} -> {preprocessed_testing}")
                clean_text_from_analysis(testing_data, preprocessed_testing, test_results)
            else:
                preprocessed_testing = None
            
            # Use the preprocessed files for training
            print(f"Preprocessing complete in {clean_duration:.2f} seconds.")
            
            # Verify the quality improved
            print("Verifying preprocessing results...")
            post_results = analyze_file(preprocessed_training)
            if post_results:
                post_quality = compute_quality_score(post_results)
                print(f"Quality score after cleaning: {post_quality:.2f}/1.0")
                
                if post_quality > quality_score:
                    print(f"Quality improved by {(post_quality - quality_score) * 100:.1f}%")
                    # Use the cleaned data
                    training_data = preprocessed_training
                    testing_data = preprocessed_testing
                else:
                    print("Warning: Cleaning did not improve quality. Using original data.")
        
        # Training phase
        print("\nProceeding to model training...")
        train_model(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_path=None,  # Not using data_path in combined mode
            dataset_path=None,  # Not using dataset_path in combined mode
            output_path=output_path,
            log_path=log_path,
            training_data=training_data,
            testing_data=testing_data,
            training_labels=None,  # No separate labels in Common Crawl
            testing_labels=None,  # No separate labels in Common Crawl
            use_gpu=args.use_gpu,
            optimizer=args.optimizer,
            use_amp=not args.no_amp
        )
        
        total_duration = time.time() - start_time
        print(f"\nComplete pipeline (dataset creation + training) finished in {total_duration:.2f} seconds")
        sys.exit(0)  # Exit after combined workflow
    
    # New mode: explicit dataset analysis
    if args.mode == 'analyze':
        from performance_analysis import analyze_file, generate_report
        
        if not args.training_data:
            print("Error: --training_data parameter is required for analysis mode")
            sys.exit(1)
            
        print(f"Analyzing dataset quality: {args.training_data}")
        results = analyze_file(training_data)
        
        if results:
            # Display results
            print("\nDataset Quality Analysis Results:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
                    
            # Generate a visual report if possible
            report_dir = os.path.join(output_path, 'reports')
            os.makedirs(report_dir, exist_ok=True)
            generate_report(results, report_dir)
            print(f"Analysis report saved to {report_dir}")
        else:
            print("Analysis failed. Check the file format and try again.")
        
        sys.exit(0)
    
    if args.mode == 'train':
        from train_model import train_model
        from performance_analysis import analyze_file, compute_quality_score, clean_text_from_analysis
        
        print("Starting training process...")
        
        # Performance tracking
        start_time = time.time()
        
        # Perform dataset quality analysis if not explicitly skipped
        should_preprocess = False
        if not args.skip_quality_check and training_data and (args.analyze_quality or args.auto_clean):
            print(f"Analyzing training data quality: {training_data}")
            results = analyze_file(training_data)
            
            if results:
                print("\nTraining Data Quality Analysis:")
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                
                # Calculate overall quality score (0.0-1.0)
                quality_score = compute_quality_score(results)
                print(f"\nOverall quality score: {quality_score:.2f}/1.0")
                
                # Decide if preprocessing is needed based on quality score
                if quality_score < args.quality_threshold:
                    print(f"Dataset quality below threshold ({args.quality_threshold})")
                    if args.auto_clean:
                        should_preprocess = True
                        print("Auto-cleaning will be performed")
                    else:
                        print("Consider using --auto_clean to automatically clean the dataset")
                else:
                    print(f"Dataset quality is good (above threshold of {args.quality_threshold})")
            else:
                print("Warning: Quality analysis failed. Proceeding without analysis.")
        
        # Preprocess data if needed and preprocessing is enabled
        if should_preprocess:
            print("\nPreprocessing data before training...")
            
            # Create a preprocessed version of the training data
            preprocessed_dir = os.path.join(output_path, 'preprocessed')
            os.makedirs(preprocessed_dir, exist_ok=True)
            
            preprocessed_training = os.path.join(preprocessed_dir, 
                                               os.path.basename(training_data))
            
            print(f"Cleaning training data: {training_data} -> {preprocessed_training}")
            start_clean_time = time.time()
            
            # Use the results from analysis to guide cleaning
            clean_text_from_analysis(training_data, preprocessed_training, results)
            
            clean_duration = time.time() - start_clean_time
            
            # If testing data exists, also preprocess it
            if testing_data:
                # Analyze testing data
                test_results = analyze_file(testing_data)
                preprocessed_testing = os.path.join(preprocessed_dir, 
                                                  os.path.basename(testing_data))
                print(f"Cleaning testing data: {testing_data} -> {preprocessed_testing}")
                clean_text_from_analysis(testing_data, preprocessed_testing, test_results)
            else:
                preprocessed_testing = None
            
            # Use the preprocessed files for training
            print(f"Preprocessing complete in {clean_duration:.2f} seconds.")
            
            # Verify the quality improved
            print("Verifying preprocessing results...")
            post_results = analyze_file(preprocessed_training)
            if post_results:
                post_quality = compute_quality_score(post_results)
                print(f"Quality score after cleaning: {post_quality:.2f}/1.0")
                
                if post_quality > quality_score:
                    print(f"Quality improved by {(post_quality - quality_score) * 100:.1f}%")
                    # Use the cleaned data
                    training_data = preprocessed_training
                    testing_data = preprocessed_testing
                else:
                    print("Warning: Cleaning did not improve quality. Using original data.")
        
        # Training phase
        print("\nProceeding to model training...")
        train_model(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_path=data_path,
            dataset_path=dataset_path,
            output_path=output_path,
            log_path=log_path,
            training_data=training_data,
            testing_data=testing_data,
            training_labels=training_labels,
            testing_labels=testing_labels,
            use_gpu=args.use_gpu,
            optimizer=args.optimizer,
            use_amp=not args.no_amp
        )
        
        total_duration = time.time() - start_time
        print(f"\nTotal training pipeline completed in {total_duration:.2f} seconds")
        
    elif args.mode == 'predict':
        if not args.text:
            print("Error: Text argument is required in predict mode")
            sys.exit(1)
            
        from predict import load_model_and_vectorizer, predict_text
        
        # Determine file extension: default to .pkl, but allow .h5 for compatibility
        model_base = os.path.splitext(args.model_file)[0]
        model_path = os.path.join(output_path, args.model_file)
        
        # Try different extensions if file not found
        if not os.path.exists(model_path):
            for ext in ['.pkl', '.h5']:
                test_path = os.path.join(output_path, model_base + ext)
                if os.path.exists(test_path):
                    model_path = test_path
                    print(f"Found model file with extension {ext}")
                    break
        
        vectorizer_path = os.path.join(output_path, args.vectorizer_file)
        label_map_path = os.path.join(output_path, "label_map.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        if not os.path.exists(vectorizer_path):
            print(f"Error: Vectorizer not found at {vectorizer_path}")
            sys.exit(1)
        
        print(f"Loading model from {model_path}")
        print(f"Loading vectorizer from {vectorizer_path}")
        
        # Check if label map exists
        if os.path.exists(label_map_path):
            print(f"Loading label map from {label_map_path}")
            model, vectorizer, label_map = load_model_and_vectorizer(model_path, vectorizer_path, label_map_path)
            
            print("Making prediction...")
            result = predict_text(args.text, model, vectorizer, label_map)
        else:
            print("No label map found. Using default label indices.")
            model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
            
            print("Making prediction...")
            result = predict_text(args.text, model, vectorizer)
        
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
    elif args.mode == 'visualize':
        from visualize_network import create_visualization
        
        # Default text if not provided
        text = args.text or "Natural language processing is fascinating"
        
        model_path = os.path.join(output_path, args.model_file)
        vectorizer_path = os.path.join(output_path, args.vectorizer_file)
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Visualization will be limited.")
        if not os.path.exists(vectorizer_path):
            print(f"Warning: Vectorizer not found at {vectorizer_path}. Visualization will be limited.")
        
        # Process words into a list
        words_list = args.vis_words.split(',') if args.vis_words else None
        
        print("Creating visualization...")
        create_visualization(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            text=text,
            output_file=args.vis_output,
            words=words_list,
            quality=args.vis_quality
        )
        print(f"Visualization saved to {args.vis_output}")
    
    elif args.mode == 'create_dataset':
        # Import modules needed for dataset creation
        from load_textdata import extract_text, process_warc
        from text_cleaning import clean_text, batch_clean_files
        
        # Handle listing available Common Crawl indexes
        if args.list_crawls:
            from data_generation import list_available_crawls
            print("Available Common Crawl indexes:")
            for crawl in list_available_crawls():
                print(f"  {crawl}")
            sys.exit(0)
            
        print(f"Creating text dataset...")
        
        # Get the output directory
        output_dir = os.path.abspath(args.output_dir)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Process based on data source (Common Crawl or local files)
        if args.common_crawl:
            # Use Common Crawl for dataset generation
            from data_generation import run_dataset_generation
            
            print("Generating dataset from Common Crawl...")
            result = run_dataset_generation(
                output_dir=output_dir,
                topics=args.topics,
                domains=args.domains,
                limit_per_query=args.limit_per_query,
                crawl_index=args.crawl_index,
                max_warcs=args.max_warcs,
                max_records=args.max_records,
                split_ratio=args.split_ratio,
                clean=args.clean,
                deep_clean=args.deep_clean,
                seed=42,
                max_workers=args.num_workers,
                batch_size=args.file_batch_size
            )
            
            if result:
                print("\nCommon Crawl dataset creation complete!")
                print(f"Training file: {result['train_file']}")
                print(f"Testing file: {result['test_file']}")
                print(f"Downloaded {len(result['downloaded_files'])} WARC files")
                print(f"Created {len(result['extracted_files'])} extracted files")
                print(f"Metadata file: {result['metadata_file']}")
                return
            else:
                print("Dataset creation failed")
                sys.exit(1)
        
        # Process based on whether clean_only is specified
        if args.clean_only:
            # Only clean existing text files
            print(f"Cleaning text files in {output_dir}...")
            batch_clean_files(output_dir, args.deep_clean, output_dir,
                             pattern=args.file_pattern.replace('.warc', '.txt'))
            return
            
        # Get the input directory (required if not clean_only or common_crawl)
        input_dir = args.input_dir
        if not input_dir:
            print("Error: Input directory is required when not using --clean_only or --common_crawl")
            sys.exit(1)
        
        input_dir = os.path.abspath(input_dir)
        if not os.path.exists(input_dir):
            print(f"Error: Input directory '{input_dir}' does not exist")
            sys.exit(1)
            
        print(f"Processing {args.file_pattern} files from {input_dir} to {output_dir}")
        
        # Get list of WARC files to process
        if args.recursive:
            warc_files = []
            for root, _, files in os.walk(input_dir):
                warc_files.extend([os.path.join(root, f) for f in files 
                                  if re.match(args.file_pattern.replace('*', '.*'), f)])
        else:
            warc_files = glob.glob(os.path.join(input_dir, args.file_pattern))
        
        # Check if we found any files
        if not warc_files:
            print(f"No files matching '{args.file_pattern}' found in {input_dir}")
            sys.exit(1)
        
        print(f"Found {len(warc_files)} files to process")
        
        # If sample option is used, select random files
        if args.sample and args.sample < len(warc_files):
            import random
            random.seed(42)  # For reproducibility
            warc_files = random.sample(warc_files, args.sample)
            print(f"Randomly selected {args.sample} files to process")
        
        # Process each WARC file
        processed_files = []
        for warc_file in warc_files:
            try:
                # Skip already processed files unless force_process is set
                base_name = os.path.basename(warc_file).replace('.warc.gz', '').replace('.warc', '')
                output_file = os.path.join(output_dir, base_name + '.txt')
                
                if os.path.exists(output_file) and not args.force_process:
                    print(f"Skipping {warc_file} (output file already exists)")
                    processed_files.append(output_file)
                    continue
                    
                print(f"Processing {warc_file}...")
                output_file = process_warc(warc_file, output_dir, 
                                          max_records=args.max_records, 
                                          num_workers=args.num_workers,
                                          batch_size=args.file_batch_size)
                
                processed_files.append(output_file)
                
                # Clean the text if requested
                if args.clean:
                    print(f"Cleaning {output_file}...")
                    clean_text(output_file, args.deep_clean, output_file)
            
            except Exception as e:
                print(f"Error processing {warc_file}: {e}")
        
        print(f"Processed {len(processed_files)} files")
        
        # Create catalog file with list of processed files
        catalog_file = os.path.join(output_dir, 'catalog.txt')
        with open(catalog_file, 'w', encoding='utf-8') as f:
            f.write(f"# Text data catalog generated on {datetime.now()}\n")
            f.write(f"# Total files: {len(processed_files)}\n\n")
            for file_path in processed_files:
                f.write(f"{os.path.basename(file_path)}\n")
        
        print(f"Created catalog file: {catalog_file}")
    
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
