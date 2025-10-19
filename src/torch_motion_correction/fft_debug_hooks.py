"""
Debug hooks to find which FFT operation is causing MKL errors during backward pass.
"""

import torch
import functools


class FFTDebugger:
    """Context manager to debug FFT operations and their gradients."""
    
    def __init__(self):
        self.original_functions = {}
        self.call_count = {}
        self.error_location = None
        
    def __enter__(self):
        """Wrap all FFT functions with debug wrappers."""
        self._wrap_fft_functions()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original FFT functions."""
        self._restore_fft_functions()
        
        if exc_type is not None:
            print("\n" + "="*70)
            print("FFT ERROR DETECTED!")
            print("="*70)
            print(f"Error type: {exc_type.__name__}")
            print(f"Error message: {exc_val}")
            print("\nFFT call history:")
            for func_name, count in self.call_count.items():
                print(f"  {func_name}: called {count} times")
            print("="*70)
        
        return False  # Re-raise the exception
    
    def _wrap_fft_functions(self):
        """Wrap all torch.fft functions with debug wrappers."""
        fft_functions = [
            'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
            'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
            'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
        ]
        
        for func_name in fft_functions:
            if hasattr(torch.fft, func_name):
                original_func = getattr(torch.fft, func_name)
                self.original_functions[func_name] = original_func
                self.call_count[func_name] = 0
                
                # Create wrapped function
                wrapped_func = self._create_debug_wrapper(func_name, original_func)
                setattr(torch.fft, func_name, wrapped_func)
    
    def _restore_fft_functions(self):
        """Restore original FFT functions."""
        for func_name, original_func in self.original_functions.items():
            setattr(torch.fft, func_name, original_func)
    
    def _create_debug_wrapper(self, func_name, original_func):
        """Create a debug wrapper for an FFT function."""
        
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            self.call_count[func_name] += 1
            call_num = self.call_count[func_name]
            
            print(f"\n🔍 FFT Call #{call_num}: {func_name}")
            
            # Check input tensor info
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                input_tensor = args[0]
                print(f"   Input device: {input_tensor.device}")
                print(f"   Input dtype: {input_tensor.dtype}")
                print(f"   Input shape: {input_tensor.shape}")
                print(f"   Input requires_grad: {input_tensor.requires_grad}")
                print(f"   Input is_leaf: {input_tensor.is_leaf}")
                print(f"   Input grad_fn: {input_tensor.grad_fn}")
                
                # Check if on CPU (which would use MKL)
                if input_tensor.device.type == 'cpu':
                    print(f"   ⚠️  WARNING: FFT on CPU will use MKL!")
            
            # Call the original function
            try:
                result = original_func(*args, **kwargs)
                print(f"   ✅ Forward pass succeeded")
                
                # Check output
                if isinstance(result, torch.Tensor):
                    print(f"   Output device: {result.device}")
                    print(f"   Output requires_grad: {result.requires_grad}")
                    
                    # Register a hook to track backward pass
                    if result.requires_grad:
                        result.register_hook(
                            self._create_backward_hook(func_name, call_num)
                        )
                
                return result
                
            except Exception as e:
                print(f"   ❌ Forward pass FAILED: {e}")
                self.error_location = f"{func_name} call #{call_num} (forward)"
                raise
        
        return wrapper
    
    def _create_backward_hook(self, func_name, call_num):
        """Create a backward hook to track gradient computation."""
        
        def hook(grad):
            print(f"\n🔙 Backward through {func_name} call #{call_num}")
            if grad is not None:
                print(f"   Gradient device: {grad.device}")
                print(f"   Gradient dtype: {grad.dtype}")
                print(f"   Gradient shape: {grad.shape}")
                
                if grad.device.type == 'cpu':
                    print(f"   ⚠️  WARNING: Gradient on CPU - will use MKL for backward FFT!")
            else:
                print(f"   ❌ Gradient is None!")
            
            return grad
        
        return hook


def debug_backward_with_fft_tracking():
    """
    Decorator to track FFT operations during forward and backward passes.
    
    Usage:
        @debug_backward_with_fft_tracking()
        def my_function():
            # code that does FFTs and backward()
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with FFTDebugger() as debugger:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def find_cpu_tensors_in_computation_graph(tensor, max_depth=20):
    """
    Recursively find all CPU tensors in the computation graph.
    This helps identify where CPU operations might be causing MKL issues.
    """
    print("\n" + "="*70)
    print("SEARCHING FOR CPU TENSORS IN COMPUTATION GRAPH")
    print("="*70)
    
    cpu_tensors = []
    visited = set()
    
    def _search_recursive(t, depth=0):
        if depth > max_depth or id(t) in visited:
            return
        
        visited.add(id(t))
        
        if not isinstance(t, torch.Tensor):
            return
        
        indent = "  " * depth
        
        # Check if on CPU
        if t.device.type == 'cpu':
            print(f"{indent}❌ Found CPU tensor at depth {depth}")
            print(f"{indent}   Shape: {t.shape}, dtype: {t.dtype}")
            print(f"{indent}   requires_grad: {t.requires_grad}")
            if t.grad_fn:
                print(f"{indent}   grad_fn: {t.grad_fn.__class__.__name__}")
            cpu_tensors.append((depth, t))
        
        # Recurse through grad_fn
        if t.grad_fn is not None and hasattr(t.grad_fn, 'next_functions'):
            for next_fn, _ in t.grad_fn.next_functions:
                if next_fn is not None and hasattr(next_fn, 'variable'):
                    _search_recursive(next_fn.variable, depth + 1)
    
    _search_recursive(tensor)
    
    print(f"\nFound {len(cpu_tensors)} CPU tensors in computation graph")
    print("="*70 + "\n")
    
    return cpu_tensors


def add_device_check_hooks(model_or_params):
    """
    Add hooks to parameters to check if gradients end up on wrong device.
    """
    def check_grad_device(grad, param_name, expected_device):
        if grad is not None:
            if grad.device != expected_device:
                print(f"\n⚠️  Device mismatch for {param_name}!")
                print(f"   Parameter device: {expected_device}")
                print(f"   Gradient device: {grad.device}")
        return grad
    
    if hasattr(model_or_params, 'named_parameters'):
        params = model_or_params.named_parameters()
    else:
        params = model_or_params
    
    for name, param in params:
        if param.requires_grad:
            expected_device = param.device
            param.register_hook(
                lambda grad, n=name, d=expected_device: check_grad_device(grad, n, d)
            )

