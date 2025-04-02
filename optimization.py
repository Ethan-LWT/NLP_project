import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class Lion(Optimizer):
    """
    Lion optimizer (Learning with Inertia and Nesterov) implementation.
    
    Based on the paper: 
    "Symbolic Discovery of Optimization Algorithms" (Google Research, 2023)
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Lion update: use sign of momentum for the update
                update = exp_avg.mul(beta2).add_(grad, alpha=1 - beta2).sign_()
                
                # Update parameters
                p.add_(update, alpha=-group['lr'])
                
        return loss


class AMPHelper:
    """Helper class for Automatic Mixed Precision training"""
    def __init__(self, enabled=True, device=None):
        self.enabled = enabled and torch.cuda.is_available()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Remove the Lion-specific message
        if self.enabled:
            try:
                # Try the new PyTorch 2.0+ API
                self.scaler = torch.amp.GradScaler(device_type='cuda')
                print("Automatic Mixed Precision (AMP) enabled")
            except (TypeError, ValueError):
                # Fallback to old API for older PyTorch versions
                try:
                    self.scaler = torch.cuda.amp.GradScaler()
                    print("Automatic Mixed Precision (AMP) enabled (legacy mode)")
                except Exception as e:
                    print(f"Warning: Failed to initialize GradScaler: {e}. AMP will be disabled.")
                    self.enabled = False
                    self.scaler = None
        else:
            self.scaler = None
            
        # Track whether we've initialized the loss scaling for proper step() calls
        self._scale_set_for_iteration = False
        
    def get_autocast_context(self):
        """Returns the autocast context manager for mixed precision training"""
        if not self.enabled:
            # Return a dummy context manager if autocast is disabled
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()
            
        try:
            # New API (PyTorch 2.0+)
            return torch.amp.autocast(device_type='cuda', enabled=self.enabled)
        except (TypeError, ValueError):
            # Fallback to old API for older PyTorch versions
            try:
                return torch.cuda.amp.autocast(enabled=self.enabled)
            except Exception as e:
                print(f"Warning: Failed to create autocast context: {e}. Using dummy context.")
                # Return a dummy context manager if autocast fails
                class DummyContext:
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                return DummyContext()
        
    def scale_loss(self, loss):
        """Scales the loss for backpropagation with mixed precision"""
        if self.enabled and self.scaler is not None:
            self._scale_set_for_iteration = True
            return self.scaler.scale(loss)
        return loss
    
    def unscale_gradients(self, optimizer):
        """Unscales gradients before clipping or modifying them"""
        if self.enabled and self.scaler is not None and self._scale_set_for_iteration:
            try:
                self.scaler.unscale_(optimizer)
            except RuntimeError as e:
                if "unscale_() has already been called" not in str(e):
                    print(f"Warning: Error unscaling gradients: {e}")
        
    def step(self, optimizer):
        """Performs optimizer step with gradient scaling"""
        try:
            if self.enabled and self.scaler is not None and self._scale_set_for_iteration:
                # Handle all optimizers consistently including Lion
                self.scaler.step(optimizer)
                self.scaler.update()
                self._scale_set_for_iteration = False
            else:
                # Direct optimizer step without scaling
                optimizer.step()
        except RuntimeError as e:
            if "unscale_" in str(e) and isinstance(optimizer, Lion):
                # Handle Lion compatibility issues with AMP by manually unscaling and stepping
                try:
                    self.unscale_gradients(optimizer)
                    optimizer.step()  # Direct step after unscaling
                    self.scaler.update()  # Still update the scaler
                except Exception as inner_e:
                    print(f"Error in Lion optimizer step: {inner_e}")
                    self._apply_gradients_to_model(optimizer)
            else:
                print(f"Error in optimizer step: {e}")
                # Manual step as fallback
                self._apply_gradients_to_model(optimizer)
    
    def _apply_gradients_to_model(self, optimizer):
        """Apply gradients manually to model parameters if optimizer step fails"""
        for group in optimizer.param_groups:
            for i, p in enumerate(group['params']):
                # Skip if no gradient is stored for this parameter
                if not hasattr(p, 'grad') or p.grad is None:
                    continue
                
                # Apply optimizer update
                if 'original_layers' in group and i < len(group['original_layers']):
                    layer = group['original_layers'][i]
                    if isinstance(optimizer, Lion):
                        # Special handling for Lion optimizer
                        if hasattr(layer, 'weights_gradient'):
                            p.data -= group['lr'] * layer.weights_gradient.sign()
                    else:
                        # Standard optimizer update
                        p.data -= group['lr'] * p.grad
                else:
                    # Fallback for parameters without layer reference
                    p.data -= group['lr'] * p.grad
        
    def zero_grad(self, optimizer):
        """Zeroes optimizer gradients"""
        optimizer.zero_grad(set_to_none=True)


def create_optimizer(model, optimizer_name='lion', lr=0.001, weight_decay=0.0):
    """
    Create an optimizer instance based on the specified name
    
    Args:
        model: NeuralNetwork model to optimize
        optimizer_name (str): Name of optimizer ('lion', 'adam', 'adamw', 'sgd')
        lr (float): Learning rate
        weight_decay (float): Weight decay for regularization
        
    Returns:
        optimizer: PyTorch optimizer instance or custom optimizer for the model
    """
    # Extract parameters from our custom neural network
    params = []
    original_layers = []
    layer_dict = {}  # Map param to originating layer
    
    for layer_idx, layer in enumerate(model.layers):
        layer_params = layer.get_parameters()
        if layer_params:
            for param_idx, param in enumerate(layer_params):
                if torch.is_tensor(param):
                    # PyTorch tensor - directly add to params
                    params.append(param)
                    original_layers.append(layer)
                    layer_dict[param] = (layer_idx, param_idx)
                elif isinstance(param, np.ndarray):
                    # NumPy array - convert to tensor
                    tensor_param = torch.tensor(param, device='cuda' if torch.cuda.is_available() else 'cpu',
                                              dtype=torch.float32, requires_grad=True)
                    params.append(tensor_param)
                    original_layers.append(layer)
                    layer_dict[tensor_param] = (layer_idx, param_idx)
    
    # If no params found or empty model, return dummy optimizer
    if not params:
        print("Warning: No parameters found for optimization")
        return DummyOptimizer(params)
    
    # Create the appropriate optimizer
    try:
        if optimizer_name.lower() == 'lion':
            optimizer = Lion(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            print(f"Unsupported optimizer: {optimizer_name}, falling back to Adam")
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        # Store references to original layers for custom gradient application
        optimizer.param_groups[0]['original_layers'] = original_layers
        optimizer.param_groups[0]['layer_dict'] = layer_dict
        return optimizer
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        return DummyOptimizer(params)

# Add a DummyOptimizer for fallback in case of errors
class DummyOptimizer:
    """Fallback optimizer that does nothing, used when regular optimizers fail"""
    def __init__(self, params):
        self.params = list(params)
        self.param_groups = [{'params': self.params, 'lr': 0.001}]
        
    def zero_grad(self, set_to_none=False):
        pass
        
    def step(self):
        pass
