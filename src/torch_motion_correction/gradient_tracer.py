"""
Gradient tracing utilities to debug where gradients are lost in the computation graph.
"""

import torch


def trace_tensor_gradient_info(tensor, name="tensor", verbose=True):
    """
    Print detailed gradient information about a tensor.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to inspect
    name : str
        Name to display for the tensor
    verbose : bool
        If True, print detailed information
        
    Returns
    -------
    dict
        Dictionary containing gradient information
    """
    info = {
        'name': name,
        'type': type(tensor).__name__,
        'requires_grad': tensor.requires_grad if isinstance(tensor, torch.Tensor) else False,
        'grad_fn': str(tensor.grad_fn) if isinstance(tensor, torch.Tensor) and tensor.grad_fn else None,
        'is_leaf': tensor.is_leaf if isinstance(tensor, torch.Tensor) else None,
        'dtype': tensor.dtype if isinstance(tensor, torch.Tensor) else None,
        'device': str(tensor.device) if isinstance(tensor, torch.Tensor) else None,
        'shape': tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
    }
    
    # Determine if differentiable
    if isinstance(tensor, torch.Tensor):
        info['is_differentiable'] = tensor.requires_grad or (tensor.grad_fn is not None)
    else:
        info['is_differentiable'] = False
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Gradient Info: {name}")
        print(f"{'='*70}")
        print(f"  Type:            {info['type']}")
        if isinstance(tensor, torch.Tensor):
            print(f"  Shape:           {info['shape']}")
            print(f"  Dtype:           {info['dtype']}")
            print(f"  Device:          {info['device']}")
            print(f"  requires_grad:   {info['requires_grad']}")
            print(f"  grad_fn:         {info['grad_fn']}")
            print(f"  is_leaf:         {info['is_leaf']}")
            print(f"  Differentiable:  {info['is_differentiable']}")
            
            if not info['is_differentiable']:
                print(f"  ❌ WARNING: {name} is NOT differentiable!")
                print(f"     Gradients cannot flow through this tensor.")
            else:
                print(f"  ✅ {name} is differentiable")
        print(f"{'='*70}\n")
    
    return info


def check_pipeline_gradients(stages):
    """
    Check gradient flow through multiple pipeline stages.
    
    Parameters
    ----------
    stages : dict
        Dictionary mapping stage names to tensors
        
    Example
    -------
    check_pipeline_gradients({
        '1. Input': input_tensor,
        '2. After motion correction': corrected,
        '3. After dose weight': dw_image,
        '4. After inspect_peaks': result,
        '5. Loss': loss,
    })
    """
    print("\n" + "="*70)
    print("PIPELINE GRADIENT FLOW CHECK")
    print("="*70)
    
    all_differentiable = True
    
    for stage_name, tensor in stages.items():
        if tensor is None:
            print(f"\n{stage_name}: ⚠️  None (skipped)")
            continue
            
        info = trace_tensor_gradient_info(tensor, stage_name, verbose=False)
        
        status = "✅" if info['is_differentiable'] else "❌"
        grad_info = f"grad_fn={info['grad_fn']}" if info['grad_fn'] else "no grad_fn"
        req_grad = f"requires_grad={info['requires_grad']}"
        
        print(f"\n{stage_name}:")
        print(f"  {status} {req_grad}, {grad_info}")
        
        if not info['is_differentiable']:
            all_differentiable = False
            print(f"     ⚠️  GRADIENT BREAK HERE! Tensors after this point won't have gradients.")
    
    print("\n" + "="*70)
    if all_differentiable:
        print("✅ All pipeline stages are differentiable!")
    else:
        print("❌ Gradient flow is broken in the pipeline!")
    print("="*70 + "\n")
    
    return all_differentiable


def trace_backward_graph(tensor, max_depth=10):
    """
    Trace the backward computation graph from a tensor.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to trace from
    max_depth : int
        Maximum depth to trace
    """
    print("\n" + "="*70)
    print("BACKWARD COMPUTATION GRAPH")
    print("="*70)
    
    def _trace_recursive(t, depth=0, visited=None):
        if visited is None:
            visited = set()
        
        if depth > max_depth or id(t) in visited:
            return
        
        visited.add(id(t))
        indent = "  " * depth
        
        if not isinstance(t, torch.Tensor):
            print(f"{indent}└─ Not a tensor: {type(t)}")
            return
        
        if t.grad_fn is None:
            if t.requires_grad:
                print(f"{indent}└─ 🌱 Leaf Variable (trainable parameter)")
            else:
                print(f"{indent}└─ ❌ Leaf Tensor (no gradients, not trainable)")
            return
        
        grad_fn_name = t.grad_fn.__class__.__name__
        print(f"{indent}└─ {grad_fn_name}")
        
        # Recursively trace next functions
        if hasattr(t.grad_fn, 'next_functions'):
            for next_fn, _ in t.grad_fn.next_functions:
                if next_fn is not None:
                    print(f"{indent}  ├─ {next_fn.__class__.__name__}")
    
    _trace_recursive(tensor)
    print("="*70 + "\n")


def check_parameters_require_grad(model_or_params, model_name="Model"):
    """
    Check if model parameters require gradients.
    
    Parameters
    ----------
    model_or_params : torch.nn.Module or iterable
        Model or iterable of (name, param) pairs
    model_name : str
        Name of the model for display
    """
    print(f"\n{'='*70}")
    print(f"PARAMETER GRADIENT CHECK: {model_name}")
    print(f"{'='*70}")
    
    if hasattr(model_or_params, 'named_parameters'):
        params = model_or_params.named_parameters()
    else:
        params = model_or_params
    
    total_params = 0
    trainable_params = 0
    
    for name, param in params:
        total_params += 1
        status = "✅" if param.requires_grad else "❌"
        
        print(f"{status} {name}:")
        print(f"     requires_grad={param.requires_grad}, shape={tuple(param.shape)}")
        
        if param.requires_grad:
            trainable_params += 1
    
    print(f"\n{'='*70}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    if trainable_params == 0:
        print("❌ WARNING: No parameters require gradients!")
    else:
        print(f"✅ {trainable_params}/{total_params} parameters are trainable")
    print(f"{'='*70}\n")


def debug_backward_hook(tensor_name):
    """
    Create a backward hook to debug gradient flow.
    
    Usage:
        tensor.register_hook(debug_backward_hook("tensor_name"))
    """
    def hook(grad):
        print(f"\n🔍 Backward hook for {tensor_name}:")
        if grad is None:
            print(f"   ❌ grad is None!")
        else:
            print(f"   ✅ grad shape: {grad.shape}")
            print(f"   ✅ grad mean: {grad.mean().item():.6f}")
            print(f"   ✅ grad std: {grad.std().item():.6f}")
            print(f"   ✅ grad min/max: [{grad.min().item():.6f}, {grad.max().item():.6f}]")
        return grad
    return hook


def add_gradient_debugging_to_pipeline(
    deformation_field,
    corrected_movie,
    dw_image,
    result,
    loss,
    parameters
):
    """
    Add comprehensive gradient debugging to the motion correction pipeline.
    
    Parameters
    ----------
    deformation_field : torch.Tensor
        The deformation field tensor
    corrected_movie : torch.Tensor
        Motion corrected movie
    dw_image : torch.Tensor
        Dose weighted image
    result : tuple or dict
        Result from core_inspect_peaks
    loss : torch.Tensor
        Loss tensor
    parameters : iterable
        Model parameters
    """
    print("\n" + "🔍"*35)
    print("COMPREHENSIVE GRADIENT DEBUGGING")
    print("🔍"*35 + "\n")
    
    # Check pipeline stages
    stages = {
        '1. Deformation Field': deformation_field,
        '2. Corrected Movie': corrected_movie,
        '3. Dose Weighted Image': dw_image,
    }
    
    # Handle result based on type
    if isinstance(result, (tuple, list)):
        stages['4. core_inspect_peaks result[0]'] = result[0] if len(result) > 0 else None
        stages['5. core_inspect_peaks result[1]'] = result[1] if len(result) > 1 else None
    elif isinstance(result, dict):
        for key, val in result.items():
            stages[f'4. core_inspect_peaks[{key}]'] = val
    else:
        stages['4. core_inspect_peaks result'] = result
    
    stages['6. Loss'] = loss
    
    # Check each stage
    check_pipeline_gradients(stages)
    
    # Check parameters
    check_parameters_require_grad(parameters, "Deformation Field Parameters")
    
    # Trace backward graph
    if isinstance(loss, torch.Tensor):
        trace_backward_graph(loss)

