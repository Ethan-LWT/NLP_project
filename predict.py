import numpy as np
import argparse
import os
import sys
import pickle

def load_model_and_vectorizer(model_path, vectorizer_path, label_map_path=None):
    """Load the trained model and vectorizer"""
    # Add project directory to path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    from text_vectorization import TextVectorizer
    from neural_network import NeuralNetwork, CategoricalCrossEntropy, Embedding, Flatten, Dense, ReLU, Softmax, Identity
    
    # Load vectorizer
    vectorizer = TextVectorizer.load(vectorizer_path)
    
    # Create a new model with the same architecture
    vocab_size = len(vectorizer.word_to_idx)
    sequence_length = vectorizer.max_sequence_length
    
    # Try to load label mapping first to determine num_classes
    label_map = None
    num_classes = None
    
    # Look for label_map.pkl in model directory if not specified
    if label_map_path is None:
        model_dir = os.path.dirname(model_path)
        default_label_map = os.path.join(model_dir, "label_map.pkl")
        if os.path.exists(default_label_map):
            label_map_path = default_label_map
            print(f"Found label map at {label_map_path}")
    
    if label_map_path and os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'rb') as f:
                label_data = pickle.load(f)
                num_classes = label_data.get('num_classes')
                # We want the reverse map (from index to original label)
                label_map = label_data.get('reverse_map')
                if not label_map and 'label_map' in label_data:
                    # Create reverse map
                    label_map = {v: k for k, v in label_data['label_map'].items()}
            print(f"Loaded label map with {num_classes} classes")
        except Exception as e:
            print(f"Error loading label map: {e}")
    
    # Determine if this is a classification model (simple architecture detection)
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    
    # Determine number of classes from the last Dense layer's weights if not provided in label map
    if num_classes is None:
        for param_set in reversed(params):
            if len(param_set) == 2:  # Dense layer has weights and bias
                weights = param_set[0]
                num_classes = weights.shape[0]
                break
    
    # Default to 2 classes if we couldn't determine
    num_classes = num_classes or 2
    
    print(f"Recreating model with {num_classes} classes")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")
    
    # Create a model with the appropriate architecture
    model = NeuralNetwork()
    model.add(Embedding(vocab_size, 100))
    model.add(Flatten())
    model.add(Dense(sequence_length * 100, 128))
    model.add(ReLU())
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dense(64, num_classes))
    
    # Use softmax for classification, identity for regression
    if num_classes == 1:
        model.add(Identity())  # Use Identity for regression (single output)
    else:
        model.add(Softmax())   # Use Softmax for classification
    
    model.set_loss(CategoricalCrossEntropy())
    
    # Load saved parameters
    model.load(model_path)
    
    if label_map_path and not label_map:
        print(f"Warning: Label map file {label_map_path} did not contain valid mapping data")
    
    return model, vectorizer, label_map

def predict_text(text, model, vectorizer, label_map=None):
    """Make predictions on text using the trained model"""
    # Vectorize the text
    sequence = vectorizer.texts_to_sequences([text])
    
    # Make prediction (ensure data is properly shaped for our model)
    prediction = model.predict(sequence.T)
    
    # Get the class with highest probability
    predicted_class_idx = np.argmax(prediction, axis=0)[0]
    confidence = prediction[predicted_class_idx, 0]
    
    # Map back to original label if mapping is available
    if label_map and predicted_class_idx in label_map:
        predicted_class = label_map[predicted_class_idx]
    else:
        predicted_class = predicted_class_idx
    
    return {
        'class': predicted_class,
        'class_index': int(predicted_class_idx),
        'confidence': float(confidence),
        'probabilities': prediction[:, 0].tolist()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with trained NLP model')
    parser.add_argument('--text', type=str, required=True, help='Text to classify')
    parser.add_argument('--model_path', type=str, default="models/final_model.pkl", 
                        help='Path to the trained model')
    parser.add_argument('--vectorizer_path', type=str, default="models/vectorizer.pkl",
                        help='Path to the vectorizer')
    parser.add_argument('--label_map_path', type=str, default="models/label_map.pkl",
                        help='Path to the label mapping file')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    model_path = os.path.abspath(args.model_path)
    vectorizer_path = os.path.abspath(args.vectorizer_path)
    label_map_path = os.path.abspath(args.label_map_path) if args.label_map_path else None
    
    # Check if model and vectorizer exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(vectorizer_path):
        print(f"Error: Vectorizer file not found at {vectorizer_path}")
        sys.exit(1)
    
    # Load model and vectorizer
    model, vectorizer, label_map = load_model_and_vectorizer(
        model_path, 
        vectorizer_path,
        label_map_path
    )
    
    # Make prediction
    result = predict_text(args.text, model, vectorizer, label_map)
    
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
