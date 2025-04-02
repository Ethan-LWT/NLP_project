import os
import argparse
import numpy as np
import pickle
import sys
import time

def train_model(model_type='classifier', epochs=10, batch_size=32, 
               data_path=None, dataset_path=None, output_path='models', log_path='logs',
               training_data=None, testing_data=None, training_labels=None, testing_labels=None,
               use_gpu=True, optimizer='lion', use_amp=True):
    """
    Train a neural network model with the specified configuration.
    
    Parameters:
    - model_type: Type of model to train (classifier or seq2seq)
    - epochs: Number of epochs to train
    - batch_size: Batch size for training
    - data_path: Path to preprocessed data file
    - dataset_path: Path to directory containing dataset_builder.py
    - output_path: Path to save models
    - log_path: Path to save logs
    - training_data: Path to training data file
    - testing_data: Path to testing data file
    - training_labels: Path to training labels file
    - testing_labels: Path to testing labels file
    - use_gpu: Whether to use GPU acceleration
    - optimizer: Optimization algorithm to use (lion, adam, adamw, sgd)
    - use_amp: Whether to use automatic mixed precision
    """
    # Add project directory to path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    from text_vectorization import prepare_data, TextVectorizer, load_direct_data
    from model_builder import EncoderDecoder
    from neural_network import check_gpu_info
    
    # Create directories if they don't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Print GPU info if available
    if use_gpu:
        print("GPU Information:")
        print(check_gpu_info())
    
    # Check which data source to use
    using_direct_files = training_data is not None
    
    # Log the data sources
    if using_direct_files:
        print(f"Using direct training data from: {training_data}")
        if testing_data:
            print(f"Using direct testing data from: {testing_data}")
        if training_labels:
            print(f"Using direct training labels from: {training_labels}")
        if testing_labels:
            print(f"Using direct testing labels from: {testing_labels}")
            
        # Verify files exist
        for path, name in [(training_data, "Training data"), 
                         (testing_data, "Testing data"), 
                         (training_labels, "Training labels"), 
                         (testing_labels, "Testing labels")]:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
    elif data_path:
        print(f"Using data from file: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
    elif dataset_path:
        print(f"Using dataset from directory: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    else:
        print("No data source specified. Will attempt to create synthetic data.")
    
    # Prepare data
    print("Preparing data...")
    if using_direct_files:
        data = load_direct_data(
            training_data=training_data,
            testing_data=testing_data,
            training_labels=training_labels,
            testing_labels=testing_labels
        )
    else:
        data = prepare_data(data_path, dataset_path)
        
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    vectorizer = data['vectorizer']
    
    # Check if we have a very small dataset and handle it specially
    if len(x_train) < 10:
        print(f"Warning: Very small training set detected ({len(x_train)} samples)")
        
        # If test set is also tiny, consider creating synthetic data for training
        if len(x_test) < 3:
            print("Warning: Test set is also very small. Training might be unstable.")
            
            # For very small datasets, use more epochs but smaller learning rate
            if epochs < 20:
                epochs = 20
                print(f"Increasing epochs to {epochs} for small dataset")
    
    # Process labels to ensure they're consecutive integers starting from 0
    unique_labels = np.unique(y_train)
    print(f"Unique labels in dataset: {unique_labels}")
    
    # If only one class is present, add a dummy sample for a second class to enable proper training
    if len(unique_labels) == 1:
        print("Only one class detected. This is a special case.")
        
        # Treat as regression problem
        label_map = {unique_labels[0]: 0}
        y_train_mapped = np.zeros_like(y_train)
        y_test_mapped = np.zeros_like(y_test)
        num_classes = 1
        
        print("Treating this as a regression task with a single output neuron.")
    else:
        # Normal classification case
        label_map = {old_label: i for i, old_label in enumerate(unique_labels)}
        
        # Map the labels to new indices
        y_train_mapped = np.array([label_map[label] for label in y_train])
        y_test_mapped = np.array([label_map[label] for label in y_test])
        num_classes = len(unique_labels)
        
        print(f"Original labels: {unique_labels}")
        print(f"Mapped to: {list(range(len(unique_labels)))}")
    
    # Get parameters
    vocab_size = len(vectorizer.word_to_idx)
    sequence_length = vectorizer.max_sequence_length
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {num_classes}")
    print(f"Training data shape: {x_train.shape}")
    
    # For very large datasets, adjust batch size and learning rate
    if len(x_train) > 100000:
        print(f"Large dataset detected ({len(x_train)} samples). Adjusting training parameters.")
        # Increase batch size for efficiency
        if batch_size < 64:
            batch_size = 64
            print(f"Increased batch size to {batch_size} for large dataset")
    
    # Use smaller batch size if dataset is very small
    actual_batch_size = min(batch_size, len(x_train))
    if actual_batch_size < batch_size:
        print(f"Warning: Reduced batch size to {actual_batch_size} due to small dataset size")
    
    # For very small datasets or seq2seq with a single class or large datasets, 
    # we might want to adjust the learning rate further
    learning_rate = 0.01
    if len(x_train) < 10 or (model_type == 'seq2seq' and num_classes == 1):
        learning_rate = 0.001
        print(f"Using reduced learning rate {learning_rate} for special case")
    elif len(x_train) > 100000:
        # For very large datasets, use slightly smaller learning rate for stability
        learning_rate = 0.005
        print(f"Using adjusted learning rate {learning_rate} for large dataset")
    
    # Build model
    print(f"Building {model_type} model...")
    model_builder = EncoderDecoder(vocab_size, use_gpu=use_gpu)
    
    # For seq2seq model with only one class, we might want a different approach
    if model_type == 'seq2seq':
        if num_classes == 1:
            print("Note: Building seq2seq model for single-class data (regression mode)")
            # We might need a different structure for regression seq2seq
            # but for now, just call the existing function that was updated to handle this case
        model = model_builder.build_seq2seq_model(sequence_length, num_classes)
    else:  # default to classifier
        model = model_builder.build_classifier_model(sequence_length, num_classes)
    
    # Print architecture info
    print("Model architecture:")
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        if hasattr(layer, 'weights'):
            print(f"  Layer {i+1}: {layer_type} - Shape: {layer.weights.shape}")
        else:
            print(f"  Layer {i+1}: {layer_type}")
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    
    # Set up validation data
    validation_data = (x_test, y_test_mapped)
    
    # Train the model with early stopping
    history = model.train(
        X=x_train,
        y=y_train_mapped,
        epochs=epochs,
        batch_size=actual_batch_size,
        learning_rate=learning_rate,
        validation_data=validation_data,
        early_stopping_patience=5 if len(x_train) >= 10 else 10,  # More patience for small datasets
        verbose=True,
        save_best=True,
        optimizer=optimizer,
        use_amp=use_amp
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test_mapped)
    print(f"Test loss: {test_loss:.4f}")
    if num_classes > 1:
        print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save the model, vectorizer, and label map
    print("Saving model and vectorizer...")
    model_path = os.path.join(output_path, "final_model.pkl")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    vectorizer_path = os.path.join(output_path, "vectorizer.pkl")
    vectorizer.save(vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")
    
    # Save the label mapping for future reference
    label_map_path = os.path.join(output_path, "label_map.pkl")
    with open(label_map_path, 'wb') as f:
        pickle.dump({
            'label_map': label_map,
            'reverse_map': {v: k for k, v in label_map.items()},
            'num_classes': num_classes,
            'unique_labels': unique_labels.tolist()
        }, f)
    print(f"Label mapping saved to {label_map_path}")
    
    # Save the training history
    history_path = os.path.join(output_path, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    # Create a simple model summary
    model_summary = {
        'type': model_type,
        'vocab_size': vocab_size,
        'sequence_length': sequence_length,
        'num_classes': num_classes,
        'training_time': training_time,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy if num_classes > 1 else "N/A",
        'layers': [type(layer).__name__ for layer in model.layers]
    }
    
    summary_path = os.path.join(output_path, "model_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Model Summary\n")
        f.write("=============\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Vocabulary Size: {vocab_size}\n")
        f.write(f"Sequence Length: {sequence_length}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        if num_classes == 1:
            f.write(f"Training Mode: Regression (single class)\n")
        f.write("\nLayer Architecture:\n")
        for i, layer in enumerate(model.layers):
            layer_name = type(layer).__name__
            f.write(f"  Layer {i+1}: {layer_name}\n")
            if hasattr(layer, 'weights'):
                f.write(f"    Shape: {layer.weights.shape}\n")
        f.write(f"\nTraining Time: {training_time:.2f} seconds\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        if num_classes > 1:
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    print(f"Model summary saved to {summary_path}")
    
    return history, model, vectorizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NLP model')
    parser.add_argument('--model_type', type=str, default='classifier', choices=['classifier', 'seq2seq'],
                        help='Type of model to train (classifier or seq2seq)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default=None, help='Path to preprocessed data (optional)')
    parser.add_argument('--dataset_path', type=str, default='download_and_create_textdata',
                       help='Path to directory containing dataset_builder.py')
    parser.add_argument('--output_path', type=str, default='models', help='Path to save models')
    parser.add_argument('--log_path', type=str, default='logs', help='Path to save logs')
    
    # Add new arguments for direct data specification
    parser.add_argument('--training_data', type=str, default=None,
                      help='Path to training data file')
    parser.add_argument('--testing_data', type=str, default=None,
                      help='Path to testing data file')
    parser.add_argument('--training_labels', type=str, default=None,
                      help='Path to training labels file (if separate from training data)')
    parser.add_argument('--testing_labels', type=str, default=None,
                      help='Path to testing labels file (if separate from testing data)')
    parser.add_argument('--save_model', type=str, default=None,
                      help='Path to save the model (overrides output_path)')
    
    # Add GPU argument
    parser.add_argument('--use_gpu', action='store_true', default=True,
                      help='Use GPU acceleration if available')
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu',
                      help='Disable GPU acceleration')
    
    # Add optimizer and AMP arguments
    parser.add_argument('--optimizer', type=str, default='lion', choices=['lion', 'adam', 'adamw', 'sgd'],
                      help='Optimizer to use for training')
    parser.add_argument('--use_amp', action='store_true', default=True,
                      help='Use automatic mixed precision for training')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                      help='Disable automatic mixed precision for training')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    dataset_path = os.path.abspath(args.dataset_path) if args.dataset_path else None
    output_path = os.path.abspath(args.output_path) if args.output_path else 'models'
    log_path = os.path.abspath(args.log_path) if args.log_path else 'logs'
    training_data = os.path.abspath(args.training_data) if args.training_data else None
    testing_data = os.path.abspath(args.testing_data) if args.testing_data else None
    training_labels = os.path.abspath(args.training_labels) if args.training_labels else None
    testing_labels = os.path.abspath(args.testing_labels) if args.testing_labels else None
    output_path = os.path.abspath(args.save_model) if args.save_model else output_path
    
    train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        dataset_path=dataset_path,
        output_path=output_path,
        log_path=log_path,
        training_data=training_data,
        testing_data=testing_data,
        training_labels=training_labels,
        testing_labels=testing_labels,
        use_gpu=args.use_gpu,
        optimizer=args.optimizer,
        use_amp=args.use_amp
    )
