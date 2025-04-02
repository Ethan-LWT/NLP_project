"""
A neural network implementation using NumPy with optional GPU acceleration.
This module provides classes for building and training neural networks
for NLP tasks with optional GPU support via PyTorch.
"""

import numpy as np
import pickle
import time
import os
from typing import List, Tuple, Dict, Any, Union, Optional, Callable

# Add GPU support
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        USE_GPU = True
        print(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("No CUDA-compatible GPU found. Using CPU.")
except ImportError:
    print("PyTorch not found. Using NumPy implementation only.")
    DEVICE = None

def to_gpu(arr):
    """Convert NumPy array to GPU tensor if GPU is available"""
    if USE_GPU and DEVICE is not None:
        # Always convert to float32 for GPU operations to ensure consistency
        if isinstance(arr, np.ndarray):
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
        
        # Use clone to avoid the warning about tensor construction
        if torch.is_tensor(arr):
            return arr.detach().clone().to(device=DEVICE, dtype=torch.float32)
        else:
            return torch.tensor(arr, device=DEVICE, dtype=torch.float32)
    return arr

def to_numpy(tensor_or_array):
    """Convert tensor to NumPy array if it's a PyTorch tensor"""
    if USE_GPU and torch.is_tensor(tensor_or_array):
        return tensor_or_array.cpu().detach().numpy()
    return tensor_or_array

class Layer:
    """Base class for neural network layers"""
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        """Forward pass"""
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass"""
        raise NotImplementedError
    
    def get_parameters(self):
        """Return layer parameters"""
        return []
    
    def set_parameters(self, params):
        """Set layer parameters"""
        pass

class Dense(Layer):
    """Fully connected layer"""
    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialize weights with He initialization (use float32 for GPU compatibility)
        self.weights = np.random.randn(output_size, input_size).astype(np.float32) * np.sqrt(2 / input_size)
        self.bias = np.zeros((output_size, 1), dtype=np.float32)
        
        # GPU tensors if available
        if USE_GPU:
            self.weights_gpu = to_gpu(self.weights)
            self.bias_gpu = to_gpu(self.bias)
    
    def forward(self, input_data):
        """Forward pass"""
        self.input = input_data
        
        if USE_GPU and torch.is_tensor(input_data):
            if not hasattr(self, 'weights_gpu'):
                self.weights_gpu = to_gpu(self.weights)
                self.bias_gpu = to_gpu(self.bias)
                
            # Ensure input data is float32 to match weights
            if input_data.dtype != self.weights_gpu.dtype:
                input_data = input_data.to(dtype=self.weights_gpu.dtype)
                
            return torch.matmul(self.weights_gpu, input_data) + self.bias_gpu
        else:
            return np.dot(self.weights, input_data) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass"""
        if USE_GPU and torch.is_tensor(output_gradient):
            # Ensure output_gradient has correct shape for bias updates
            if output_gradient.dim() == 1:
                output_gradient = output_gradient.reshape(-1, 1)
            elif output_gradient.shape[1] > 1:
                # If we have multiple samples, take mean of gradients
                output_gradient_for_bias = torch.mean(output_gradient, dim=1, keepdim=True)
            else:
                output_gradient_for_bias = output_gradient
                
            weights_gradient = torch.matmul(output_gradient, self.input.t())
            input_gradient = torch.matmul(self.weights_gpu.t(), output_gradient)
            
            # Update parameters only if learning_rate is provided (not when using optimizer)
            if learning_rate is not None:
                self.weights_gpu -= learning_rate * weights_gradient
                self.bias_gpu -= learning_rate * output_gradient_for_bias
                
                # Update CPU weights too for saving/loading
                self.weights = to_numpy(self.weights_gpu)
                self.bias = to_numpy(self.bias_gpu)
            
            # Store gradients for optimizer
            self.weights_gradient = weights_gradient
            self.bias_gradient = output_gradient_for_bias
            
            return input_gradient
        else:
            # Ensure output_gradient has correct shape for bias updates
            if output_gradient.ndim == 1:
                output_gradient = output_gradient.reshape(-1, 1)
            elif output_gradient.shape[1] > 1:
                # If we have multiple samples, take mean of gradients
                output_gradient_for_bias = np.mean(output_gradient, axis=1, keepdims=True)
            else:
                output_gradient_for_bias = output_gradient
            
            weights_gradient = np.dot(output_gradient, self.input.T)
            input_gradient = np.dot(self.weights.T, output_gradient)
            
            # Update parameters only if learning_rate is provided
            if learning_rate is not None:
                self.weights -= learning_rate * weights_gradient
                self.bias -= learning_rate * output_gradient_for_bias
            
            # Store gradients for optimizer
            self.weights_gradient = weights_gradient
            self.bias_gradient = output_gradient_for_bias
            
            return input_gradient
    
    def get_parameters(self):
        """Return layer parameters"""
        return [self.weights, self.bias]
    
    def set_parameters(self, params):
        """Set layer parameters"""
        self.weights = params[0]
        self.bias = params[1]

class Embedding(Layer):
    """Embedding layer for text input"""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Initialize with small random values (use float32 for GPU compatibility)
        self.weights = np.random.randn(embedding_dim, vocab_size).astype(np.float32) * 0.01
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # GPU tensors if available
        if USE_GPU:
            self.weights_gpu = to_gpu(self.weights)
    
    def forward(self, input_data):
        """
        Forward pass
        Input: batch of sequences of word indices (batch_size, sequence_length)
        Output: embedded vectors (embedding_dim * sequence_length, batch_size)
        """
        # GPU implementation for forward pass
        if USE_GPU and torch.is_tensor(input_data):
            self.original_input = input_data
            
            batch_size = input_data.shape[1] if input_data.dim() > 1 else 1
            sequence_length = input_data.shape[0]
            
            self.batch_size = batch_size
            self.sequence_length = sequence_length
            
            if not hasattr(self, 'weights_gpu'):
                self.weights_gpu = to_gpu(self.weights)
            
            # Create tensor with consistent dtype (float32)
            embedded = torch.zeros(
                self.embedding_dim * sequence_length, 
                batch_size, 
                device=DEVICE,
                dtype=torch.float32
            )
            
            # Store input tensor for backward pass - IMPORTANT FIX
            self.input = input_data
            
            for i in range(batch_size):
                for j in range(sequence_length):
                    word_idx = int(input_data[j, i].item())
                    if word_idx >= self.vocab_size:
                        word_idx = 0  # Use padding token
                    embedded[j*self.embedding_dim:(j+1)*self.embedding_dim, i] = self.weights_gpu[:, word_idx]
            
            return embedded
        else:
            self.original_input = input_data
            
            batch_size = input_data.shape[1] if input_data.ndim > 1 else 1
            
            if input_data.ndim == 1:  # Single sample
                sequence_length = input_data.shape[0]
                input_data = input_data.reshape(1, sequence_length)
            elif input_data.ndim == 2 and input_data.shape[0] != batch_size:
                input_data = input_data.T
            
            batch_size, sequence_length = input_data.shape
            self.batch_size = batch_size
            self.sequence_length = sequence_length
            
            self.input = input_data.copy()
            self.input_shape = input_data.shape
            
            embedded = np.zeros((self.embedding_dim * sequence_length, batch_size))
            
            for i in range(batch_size):
                for j in range(sequence_length):
                    word_idx = input_data[i, j]
                    if word_idx >= self.vocab_size:
                        word_idx = 0
                    embedded[j*self.embedding_dim:(j+1)*self.embedding_dim, i] = self.weights[:, word_idx]
            
            return embedded
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass with GPU support"""
        if USE_GPU and torch.is_tensor(output_gradient):
            batch_size = self.batch_size
            sequence_length = self.sequence_length
            
            if output_gradient.shape[1] != batch_size:
                print(f"Warning: Gradient batch size ({output_gradient.shape[1]}) doesn't match input batch size ({batch_size})")
                batch_size = min(batch_size, output_gradient.shape[1])
            
            weights_gradient = torch.zeros_like(self.weights_gpu)
            
            try:
                # Reshape gradient
                grad_reshaped = output_gradient.reshape(self.embedding_dim, sequence_length, -1)
                actual_batch_size = min(batch_size, grad_reshaped.shape[2])
                
                # Verify self.input exists and is a tensor
                if self.input is None or not torch.is_tensor(self.input):
                    print("Warning: Input tensor is missing or invalid in backward pass. Using zeros.")
                    # Create a dummy tensor with zeros to avoid errors
                    self.input = torch.zeros((actual_batch_size, sequence_length), 
                                            dtype=torch.long, 
                                            device=DEVICE)
                
                # Make sure we're accessing elements correctly
                if self.input.dim() != 2:
                    # If input is not 2D (e.g., it's transposed from what we expect)
                    # We need to reshape it to the expected format
                    print(f"Warning: Input tensor has unexpected shape {self.input.shape}. Attempting to reshape.")
                    if self.input.dim() == 1:
                        self.input = self.input.reshape(1, -1)
                    elif self.input.shape[0] == sequence_length and self.input.shape[1] == batch_size:
                        # Input is likely transposed - swap dimensions
                        self.input = self.input.T
                
                # Update embeddings
                for i in range(actual_batch_size):
                    for j in range(sequence_length):
                        try:
                            # Use safer access method for indices
                            if i < self.input.shape[0] and j < self.input.shape[1]:
                                word_idx = int(self.input[i, j].item())
                                if word_idx >= self.vocab_size:
                                    word_idx = 0
                                weights_gradient[:, word_idx] += grad_reshaped[:, j, i]
                        except Exception as e:
                            # More detailed error reporting
                            print(f"Warning: Error accessing index i={i}, j={j} in tensor of shape {self.input.shape}")
                            print(f"Exception details: {str(e)}")
                
                # Update weights only if learning rate is provided (not when using optimizer)
                if learning_rate is not None:
                    self.weights_gpu -= learning_rate * weights_gradient
                    self.weights = to_numpy(self.weights_gpu)
                
                # Store gradient for optimizer
                self.weights_gradient = weights_gradient
                
            except Exception as e:
                # Better error message with more context
                print(f"Error in GPU Embedding backward pass: {e}")
                print(f"Input shape: {self.input.shape if hasattr(self, 'input') and self.input is not None else 'None'}")
                print(f"Gradient shape: {output_gradient.shape}")
                print(f"Batch size: {batch_size}, Sequence length: {sequence_length}")
            
            return None
        else:
            batch_size = self.batch_size
            sequence_length = self.sequence_length
            
            weights_gradient = np.zeros_like(self.weights)
            
            if output_gradient.shape[1] != batch_size:
                print(f"Warning: Gradient batch size ({output_gradient.shape[1]}) doesn't match input batch size ({batch_size})")
                batch_size = min(batch_size, output_gradient.shape[1])
            
            try:
                grad_reshaped = output_gradient.reshape(self.embedding_dim, sequence_length, -1)
                actual_batch_size = min(batch_size, grad_reshaped.shape[2])
                
                for i in range(actual_batch_size):
                    for j in range(sequence_length):
                        try:
                            word_idx = self.input[i, j]
                            if word_idx >= self.vocab_size:
                                word_idx = 0
                            weights_gradient[:, word_idx] += grad_reshaped[:, j, i]
                        except IndexError as e:
                            print(f"Warning: Skipping invalid index at position i={i}, j={j}")
                
                # Update weights only if learning rate is provided
                if learning_rate is not None:
                    self.weights -= learning_rate * weights_gradient
                
                # Store gradient for optimizer
                self.weights_gradient = weights_gradient
                
            except Exception as e:
                print(f"Error in Embedding backward pass: {e}")
            
            return None
    
    def get_parameters(self):
        """Return layer parameters"""
        return [self.weights]
    
    def set_parameters(self, params):
        """Set layer parameters"""
        self.weights = params[0]
        if USE_GPU:
            self.weights_gpu = to_gpu(self.weights)

class Flatten(Layer):
    """Flatten layer to convert multi-dimensional input to 1D"""
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient

class Activation(Layer):
    """Base class for activation layers"""
    def __init__(self, activation, activation_gradient):
        super().__init__()
        self.activation = activation
        self.activation_gradient = activation_gradient
    
    def forward(self, input_data):
        self.input = input_data
        return self.activation(input_data)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_gradient(self.input)

class ReLU(Activation):
    """ReLU activation layer"""
    def __init__(self):
        def relu(x):
            if USE_GPU and torch.is_tensor(x):
                return torch.maximum(torch.tensor(0.0, device=DEVICE, dtype=x.dtype), x)
            return np.maximum(0, x)
        
        def relu_gradient(x):
            if USE_GPU and torch.is_tensor(x):
                return torch.where(x > 0, torch.tensor(1.0, device=DEVICE, dtype=x.dtype), 
                                  torch.tensor(0.0, device=DEVICE, dtype=x.dtype))
            return np.where(x > 0, 1, 0)
        
        super().__init__(relu, relu_gradient)

class Sigmoid(Activation):
    """Sigmoid activation layer"""
    def __init__(self):
        def sigmoid(x):
            if USE_GPU and torch.is_tensor(x):
                return 1 / (1 + torch.exp(-torch.clamp(x, -15, 15)))
            return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
        
        def sigmoid_gradient(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_gradient)

class Softmax(Layer):
    """Softmax activation layer"""
    def forward(self, input_data):
        self.input = input_data
        
        if USE_GPU and torch.is_tensor(input_data):
            # PyTorch implementation
            exp_values = torch.exp(input_data - torch.max(input_data, dim=0, keepdim=True).values)
            return exp_values / torch.sum(exp_values, dim=0, keepdim=True)
        else:
            # NumPy implementation
            exp_values = np.exp(input_data - np.max(input_data, axis=0, keepdims=True))
            return exp_values / np.sum(exp_values, axis=0, keepdims=True)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient

class Identity(Layer):
    """Identity activation layer (used for regression)"""
    def forward(self, input_data):
        self.input = input_data
        return input_data
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient

class Loss:
    """Base class for loss functions"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError

class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy loss"""
    def forward(self, y_pred, y_true):
        # Convert to CPU numpy arrays if tensors
        if USE_GPU and torch.is_tensor(y_pred):
            y_pred_np = to_numpy(y_pred)
            if torch.is_tensor(y_true):
                y_true_np = to_numpy(y_true)
            else:
                y_true_np = y_true
        else:
            y_pred_np = y_pred
            y_true_np = y_true
            
        if len(y_true_np.shape) == 1:
            batch_size = y_true_np.shape[0]
            num_classes = y_pred_np.shape[0]
            
            # Create one-hot representation of labels
            y_true_one_hot = np.zeros((num_classes, batch_size))
            
            # Ensure indices are valid integers
            valid_indices = np.clip(y_true_np, 0, num_classes-1).astype(np.int32)
            
            # Set one-hot values
            for i in range(batch_size):
                y_true_one_hot[valid_indices[i], i] = 1
                
            y_true_np = y_true_one_hot
            
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred_np, 1e-15, 1 - 1e-15)
        
        # Calculate cross entropy loss
        return -np.sum(y_true_np * np.log(y_pred_clipped)) / y_pred_np.shape[1]
    
    def backward(self, y_pred, y_true):
        # For GPU computation
        if USE_GPU and torch.is_tensor(y_pred):
            if not torch.is_tensor(y_true):
                y_true = torch.tensor(y_true, device=DEVICE, dtype=torch.float32)
                
            if y_pred.shape[0] == 1:
                if len(y_true.shape) == 1:
                    y_true = y_true.reshape(1, -1)
                return 2 * (y_pred - y_true) / y_pred.shape[1]
            
            if len(y_true.shape) == 1:
                batch_size = y_true.shape[0]
                num_classes = y_pred.shape[0]
                y_true_one_hot = torch.zeros((num_classes, batch_size), device=DEVICE, dtype=torch.float32)
                
                # Ensure indices are valid integers (PyTorch indexing requires long)
                valid_indices = torch.clamp(y_true, 0, num_classes-1).long()
                
                # Set one-hot values
                for i in range(batch_size):
                    idx = valid_indices[i].item()  # Get integer value for indexing
                    y_true_one_hot[idx, i] = 1
                
                y_true = y_true_one_hot
            
            # Gradient of cross-entropy with softmax is (y_pred - y_true)
            return -(y_true - y_pred) / y_pred.shape[1]
        
        # For CPU computation (original code)
        if y_pred.shape[0] == 1:
            if len(y_true.shape) == 1:
                y_true = y_true.reshape(1, -1)
            return 2 * (y_pred - y_true) / y_pred.shape[1]
        
        if len(y_true.shape) == 1:
            batch_size = y_true.shape[0]
            num_classes = y_pred.shape[0]
            y_true_one_hot = np.zeros((num_classes, batch_size))
            
            # Ensure indices are integers
            valid_indices = np.clip(y_true, 0, num_classes-1).astype(int)
            
            # Set one-hot values
            for i in range(batch_size):
                y_true_one_hot[valid_indices[i], i] = 1
            
            y_true = y_true_one_hot
        
        # Gradient of cross-entropy with softmax is (y_pred - y_true)
        return -(y_true - y_pred) / y_pred.shape[1]

class NeuralNetwork:
    """Neural network model with optional GPU acceleration"""
    def __init__(self, use_gpu=None):
        self.layers = []
        self.loss = None
        self.use_gpu = USE_GPU if use_gpu is None else use_gpu
    
    def add(self, layer):
        self.layers.append(layer)
    
    def set_loss(self, loss):
        self.loss = loss
    
    def predict(self, X):
        if self.use_gpu:
            X = to_gpu(X)
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return to_numpy(output) if self.use_gpu and torch.is_tensor(output) else output
    
    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.01, 
              validation_data=None, early_stopping_patience=None,
              verbose=True, save_best=None, optimizer='lion', use_amp=True):
        """Train the network with GPU acceleration if available"""
        if self.loss is None:
            raise Exception("Loss function not set")
        
        # Initialize optimizer and AMP components
        from optimization import create_optimizer, AMPHelper
        
        # Enable AMP for all optimizers including Lion - remove the special case handling
        if self.use_gpu and torch.cuda.is_available():
            amp_helper = AMPHelper(enabled=use_amp)
            optimizer_obj = create_optimizer(self, optimizer_name=optimizer, lr=learning_rate)
        else:
            # Fallback to standard training if GPU not available
            amp_helper = None
            optimizer_obj = create_optimizer(self, optimizer_name=optimizer, lr=learning_rate)
        
        if self.use_gpu:
            # Convert input data to float32 for GPU
            X = to_gpu(X)
            y = to_gpu(y)
            if validation_data is not None:
                validation_data = (to_gpu(validation_data[0]), to_gpu(validation_data[1]))
        
        num_samples = X.shape[0]
        history = {"loss": [], "accuracy": []}
        
        if validation_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []
        
        best_val_accuracy = 0
        patience_counter = 0
        best_params = None
        
        # Determine the AMP strategy
        using_amp = amp_helper is not None and amp_helper.enabled and torch.cuda.is_available()
        
        for epoch in range(epochs):
            # Generate random indices for shuffling
            if USE_GPU and torch.is_tensor(X):
                indices = torch.randperm(num_samples, device=DEVICE)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                indices = np.random.permutation(num_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            
            batch_losses = []
            batch_accuracies = []
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Reset gradients
                optimizer_obj.zero_grad(set_to_none=True)
                
                # Use unified training path with AMP if enabled
                if using_amp:
                    # Forward pass with mixed precision
                    with amp_helper.get_autocast_context():
                        # Forward pass
                        y_pred = self.predict(X_batch.T)
                        
                        # If we're using GPU but y_pred was converted to numpy, convert back to tensor
                        if self.use_gpu and not torch.is_tensor(y_pred) and torch.is_tensor(y_batch):
                            y_pred = to_gpu(y_pred)
                        
                        # Calculate loss
                        loss_value = self.loss.forward(y_pred, y_batch)
                        if torch.is_tensor(loss_value):
                            # Scale the loss for AMP training
                            scaled_loss = amp_helper.scale_loss(loss_value)
                            
                            # Backward pass on scaled loss
                            if torch.is_tensor(scaled_loss) and hasattr(scaled_loss, 'backward'):
                                scaled_loss.backward()
                            else:
                                # Manual backward pass if automatic differentiation failed
                                grad = self.loss.backward(y_pred, y_batch)
                                for layer in reversed(self.layers):
                                    grad = layer.backward(grad, None)
                        else:
                            # Handle non-tensor loss (rare case)
                            grad = self.loss.backward(y_pred, y_batch)
                            for layer in reversed(self.layers):
                                grad = layer.backward(grad, None)
                            
                    # Step optimizer with AMP-aware handling
                    amp_helper.step(optimizer_obj)
                else:
                    # Standard training path (no AMP)
                    y_pred = self.predict(X_batch.T)
                    
                    # If we're using GPU but y_pred was converted to numpy, convert back to tensor
                    if self.use_gpu and not torch.is_tensor(y_pred) and torch.is_tensor(y_batch):
                        y_pred = to_gpu(y_pred)
                    
                    # Calculate loss
                    loss_value = self.loss.forward(y_pred, y_batch)
                    
                    # Backward pass
                    grad = self.loss.backward(y_pred, y_batch)
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad, None)
                    
                    # Take optimizer step
                    optimizer_obj.step()
                
                # Store batch metrics
                if isinstance(loss_value, torch.Tensor):
                    batch_losses.append(loss_value.item())
                else:
                    batch_losses.append(float(loss_value))
                
                # Calculate accuracy
                if USE_GPU and torch.is_tensor(y_pred):
                    if len(y_batch.shape) == 1:
                        predictions = torch.argmax(y_pred, dim=0)
                        accuracy = torch.mean((predictions == y_batch).float()).item()
                    else:
                        accuracy = torch.mean((torch.argmax(y_pred, dim=0) == torch.argmax(y_batch, dim=1)).float()).item()
                else:
                    if len(y_batch.shape) == 1:
                        predictions = np.argmax(y_pred, axis=0)
                        accuracy = np.mean(predictions == y_batch)
                    else:
                        accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y_batch, axis=1))
                
                batch_accuracies.append(accuracy)
            
            # Calculate epoch metrics
            epoch_loss = np.mean(batch_losses)
            epoch_accuracy = np.mean(batch_accuracies)
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}", end="")
            
            # Validation logic remains the same
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val.T)
                
                # If we're using GPU but y_val_pred was converted to numpy, convert back to tensor
                if self.use_gpu and not torch.is_tensor(y_val_pred) and torch.is_tensor(y_val):
                    y_val_pred = to_gpu(y_val_pred)
                
                val_loss = self.loss.forward(y_val_pred, y_val)
                
                if USE_GPU and torch.is_tensor(y_val_pred):
                    val_accuracy = torch.mean((torch.argmax(y_val_pred, dim=0) == y_val).float()).item()
                else:
                    val_accuracy = np.mean(np.argmax(y_val_pred, axis=0) == y_val)
                
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)
                
                if verbose:
                    print(f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    if save_best:
                        best_params = [layer.get_parameters() for layer in self.layers]
                else:
                    patience_counter += 1
                
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            elif verbose:
                print("")  # Newline after epoch metrics
        
        # Restore best model if using early stopping
        if save_best and best_params:
            for layer, params in zip(self.layers, best_params):
                layer.set_parameters(params)
        
        return history
    
    def evaluate(self, X, y):
        """Evaluate the model on test data"""
        if self.use_gpu:
            X = to_gpu(X)
            y = to_gpu(y)
        
        y_pred = self.predict(X.T)
        
        # If we're using GPU but y_pred was converted to numpy, convert back to tensor
        if self.use_gpu and not torch.is_tensor(y_pred) and torch.is_tensor(y):
            y_pred = to_gpu(y_pred)
        
        loss = self.loss.forward(y_pred, y)
        
        if USE_GPU and torch.is_tensor(y_pred):
            accuracy = torch.mean((torch.argmax(y_pred, dim=0) == y).float()).item()
        else:
            accuracy = np.mean(np.argmax(y_pred, axis=0) == y)
        
        return loss, accuracy

    def save(self, filepath):
        """Save the neural network model to a file"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Prepare network data to save
        network_data = {
            'layers': [],
            'use_gpu': self.use_gpu
        }
        
        # Save parameters from each layer
        for i, layer in enumerate(self.layers):
            layer_data = {
                'type': layer.__class__.__name__,
                'params': layer.get_parameters()
            }
            
            # Convert any PyTorch tensors to numpy arrays
            for j, param in enumerate(layer_data['params']):
                if USE_GPU and torch.is_tensor(param):
                    layer_data['params'][j] = to_numpy(param)
                    
            # Add any specific layer attributes that might be needed
            if isinstance(layer, Embedding):
                layer_data['vocab_size'] = layer.vocab_size
                layer_data['embedding_dim'] = layer.embedding_dim
            elif isinstance(layer, Dense):
                layer_data['input_size'] = layer.weights.shape[1]
                layer_data['output_size'] = layer.weights.shape[0]
            
            network_data['layers'].append(layer_data)
        
        # Save the loss function type if available
        if self.loss is not None:
            network_data['loss'] = self.loss.__class__.__name__
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(network_data, f)
        
        return True
    
    @classmethod
    def load(cls, filepath):
        """Load a neural network model from a file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            network_data = pickle.load(f)
        
        # Create a new network instance
        model = cls(use_gpu=network_data.get('use_gpu', USE_GPU))
        
        # Create and add each layer
        for layer_data in network_data['layers']:
            layer_type = layer_data['type']
            
            # Reconstruct the layer based on its type
            if layer_type == 'Embedding':
                layer = Embedding(
                    vocab_size=layer_data['vocab_size'],
                    embedding_dim=layer_data['embedding_dim']
                )
                layer.weights = layer_data['params'][0]
                if USE_GPU:
                    layer.weights_gpu = to_gpu(layer.weights)
            
            elif layer_type == 'Dense':
                layer = Dense(
                    input_size=layer_data['input_size'],
                    output_size=layer_data['output_size']
                )
                layer.weights = layer_data['params'][0]
                layer.bias = layer_data['params'][1]
                if USE_GPU:
                    layer.weights_gpu = to_gpu(layer.weights)
                    layer.bias_gpu = to_gpu(layer.bias)
            
            elif layer_type == 'ReLU':
                layer = ReLU()
            
            elif layer_type == 'Sigmoid':
                layer = Sigmoid()
            
            elif layer_type == 'Softmax':
                layer = Softmax()
            
            elif layer_type == 'Identity':
                layer = Identity()
            
            elif layer_type == 'Flatten':
                layer = Flatten()
            
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            model.add(layer)
        
        # Set the loss function
        if 'loss' in network_data:
            loss_type = network_data['loss']
            if loss_type == 'CategoricalCrossEntropy':
                model.set_loss(CategoricalCrossEntropy())
        
        return model

def create_text_classifier(vocab_size, sequence_length, num_classes, embedding_dim=100, use_gpu=None):
    model = NeuralNetwork(use_gpu=use_gpu)
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Flatten())
    model.add(Dense(embedding_dim * sequence_length, 64))
    model.add(ReLU())
    model.add(Dense(64, num_classes))
    model.add(Softmax())
    model.set_loss(CategoricalCrossEntropy())
    return model

def create_seq2seq_model(vocab_size, sequence_length, num_classes, embedding_dim=100, use_gpu=None):
    """Create a sequence-to-sequence model with optional GPU support"""
    model = NeuralNetwork(use_gpu=use_gpu)
    
    # Adjust embedding dimension based on data size
    if sequence_length > 100:
        embedding_dim = min(embedding_dim, 50)  # Reduce embedding dim for very long sequences
    
    # Add layers with adjusted dimensions
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Flatten())  # Output: (embedding_dim * sequence_length, batch_size)
    
    # For large datasets, use larger hidden layers
    hidden_size1 = 256
    hidden_size2 = 128
    
    model.add(Dense(embedding_dim * sequence_length, hidden_size1))
    model.add(ReLU())
    model.add(Dense(hidden_size1, hidden_size2))
    model.add(ReLU())
    
    # For seq2seq with only one class (regression mode), don't use Softmax
    if num_classes == 1:
        model.add(Dense(hidden_size2, sequence_length))
        model.add(Identity())  # Use Identity activation for regression
    else:
        output_size = sequence_length * num_classes
        print(f"Creating seq2seq output layer with dimensions: {hidden_size2} -> {output_size}")
        model.add(Dense(hidden_size2, output_size))
        model.add(Softmax())
    
    # Set loss function
    model.set_loss(CategoricalCrossEntropy())
    
    return model

def make_batches(data, batch_size):
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

def check_gpu_info():
    if not USE_GPU:
        return "GPU acceleration disabled or unavailable"
    
    info = []
    info.append(f"GPU: {torch.cuda.get_device_name(0)}")
    info.append(f"CUDA Version: {torch.version.cuda}")
    
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    
    info.append(f"Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved, {mem_total:.1f}MB total")
    return "\n".join(info)
