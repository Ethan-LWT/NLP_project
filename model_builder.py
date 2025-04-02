"""
Model builder for NLP tasks using neural network implementation.
"""

import os
import numpy as np
import sys

# Add the current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from neural_network import create_text_classifier, create_seq2seq_model, NeuralNetwork, USE_GPU

class EncoderDecoder:
    """
    Model builder for encoder-decoder architecture.
    This class provides methods for building different types of models.
    """
    
    def __init__(self, vocab_size, use_gpu=None):
        self.vocab_size = vocab_size
        self.use_gpu = USE_GPU if use_gpu is None else use_gpu
    
    def build_classifier_model(self, sequence_length, num_classes):
        """
        Build a text classifier model
        
        Args:
            sequence_length (int): Maximum sequence length
            num_classes (int): Number of classes
            
        Returns:
            model: A neural network model for text classification
        """
        return create_text_classifier(
            vocab_size=self.vocab_size,
            sequence_length=sequence_length,
            num_classes=num_classes,
            use_gpu=self.use_gpu
        )
    
    def build_seq2seq_model(self, sequence_length, num_classes):
        """
        Build a sequence-to-sequence model
        
        Args:
            sequence_length (int): Maximum sequence length
            num_classes (int): Number of output classes/tokens
            
        Returns:
            model: A neural network model for sequence-to-sequence tasks
        """
        # Pass use_gpu parameter here
        return create_seq2seq_model(
            vocab_size=self.vocab_size,
            sequence_length=sequence_length,
            num_classes=num_classes,
            embedding_dim=100,  # Add explicit embedding_dim parameter
            use_gpu=self.use_gpu  # Add use_gpu parameter
        )
