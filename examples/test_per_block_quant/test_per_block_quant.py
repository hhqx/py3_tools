import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from copy import deepcopy
from typing import Tuple, Union

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """A simple MLP model for testing quantization."""
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.activation(self.linear(x))
        x = self.output(x)
        return x


class LinearPerBlockQuant(nn.Module):
    """Linear layer with per-block quantization capabilities."""
    def __init__(self, original_layer, block_size: Union[int, Tuple[int, int]]=4, 
                 w_bits=8, a_bits=8):
        super(LinearPerBlockQuant, self).__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Handle block_size as either int or tuple
        if isinstance(block_size, int):
            self.out_block_size = block_size
            self.in_block_size = block_size
        elif isinstance(block_size, tuple) and len(block_size) == 2:
            self.out_block_size, self.in_block_size = block_size
        else:
            raise ValueError("block_size must be an int or a tuple of two integers")
        
        # Copy parameters from original layer
        self.weight = nn.Parameter(original_layer.weight.clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.clone())
        else:
            self.register_parameter('bias', None)
        
        # Quantization settings
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.mode = 'float'  # 'float', 'fake_quant', 'calibration'
        
        # Calculate number of blocks in each dimension
        self.num_blocks_in_fea = (self.in_features + self.in_block_size - 1) // self.in_block_size
        self.num_blocks_out_fea = (self.out_features + self.out_block_size - 1) // self.out_block_size
        
        # Check if dimensions are evenly divisible by block sizes
        if self.in_features % self.in_block_size != 0:
            logger.warning(f"Input features ({self.in_features}) not evenly divisible "
                          f"by in_block_size ({self.in_block_size})")
        if self.out_features % self.out_block_size != 0:
            logger.warning(f"Output features ({self.out_features}) not evenly divisible "
                          f"by out_block_size ({self.out_block_size})")
        
        # Quantization parameters for weights and activations (per block)
        self.register_buffer('w_scales', torch.ones(self.out_features, self.num_blocks_in_fea))
        self.register_buffer('w_zeros', torch.zeros(self.out_features, self.num_blocks_in_fea))
        self.register_buffer('a_scales', torch.ones(self.num_blocks_in_fea))
        self.register_buffer('a_zeros', torch.zeros(self.num_blocks_in_fea))
        
        # Quantization range
        self.w_qmin = 0
        self.w_qmax = 2**w_bits - 1
        self.a_qmin = 0
        self.a_qmax = 2**a_bits - 1
    
    def calibrate_weight(self):
        """Calculate scaling factors for weights per block."""
        # Process each output row
        for out_idx in range(self.out_features):
            # Process each block within this row
            for block_idx in range(self.num_blocks_in_fea):
                start_idx = block_idx * self.in_block_size
                end_idx = min(start_idx + self.in_block_size, self.in_features)
                
                # Extract the block
                block = self.weight[out_idx, start_idx:end_idx]
                
                # Calculate min and max for this block
                w_min = block.min()
                w_max = block.max()
                
                # Calculate scale and zero point
                if w_max > w_min:
                    scale = (w_max - w_min) / (self.w_qmax - self.w_qmin)
                    zero = self.w_qmin - w_min / scale
                else:
                    scale = torch.tensor(1.0)
                    zero = 0
                
                # Store the values
                self.w_scales[out_idx, block_idx] = scale
                self.w_zeros[out_idx, block_idx] = zero
    
    def calibrate_activation(self, x):
        """Calculate scaling factors for activations per block."""
        if self.mode != 'calibration' or self.a_bits >= 16:
            return x
        
        # Process each block of activations
        for block_idx in range(self.num_blocks_in_fea):
            start_idx = block_idx * self.in_block_size
            end_idx = min(start_idx + self.in_block_size, self.in_features)
            
            # Extract block across all batch samples
            block = x[:, start_idx:end_idx]
            
            # Calculate min and max
            a_min = block.min()
            a_max = block.max()
            
            # Calculate scale and zero point
            if a_max > a_min:
                scale = (a_max - a_min) / (self.a_qmax - self.a_qmin)
                zero = self.a_qmin - a_min / scale
            else:
                scale = torch.tensor(1.0)
                zero = 0
            
            # Store values
            self.a_scales[block_idx] = scale
            self.a_zeros[block_idx] = zero
        
        return x
    
    def quantize_weight(self):
        """Apply fake quantization to weights."""
        quantized_weight = self.weight.clone()
        
        # Process each output row
        for out_idx in range(self.out_features):
            # Process each block within this row
            for block_idx in range(self.num_blocks_in_fea):
                start_idx = block_idx * self.in_block_size
                end_idx = min(start_idx + self.in_block_size, self.in_features)
                
                # Extract the block
                block = self.weight[out_idx, start_idx:end_idx]
                
                # Get scale and zero point
                scale = self.w_scales[out_idx, block_idx]
                zero = self.w_zeros[out_idx, block_idx]
                
                # Quantize and dequantize (fake quantization)
                q_block = torch.clamp(torch.round(block / scale + zero), self.w_qmin, self.w_qmax)
                dq_block = (q_block - zero) * scale
                
                # Store back
                quantized_weight[out_idx, start_idx:end_idx] = dq_block
        
        return quantized_weight
    
    def quantize_activation(self, x):
        """Apply fake quantization to activations."""
        if self.mode != 'fake_quant' or self.a_bits >= 16:
            return x
        
        quantized_x = x.clone()
        
        # Process each block
        for block_idx in range(self.num_blocks_in_fea):
            start_idx = block_idx * self.in_block_size
            end_idx = min(start_idx + self.in_block_size, self.in_features)
            
            # Extract block
            block = x[:, start_idx:end_idx]
            
            # Get scale and zero point
            scale = self.a_scales[block_idx]
            zero = self.a_zeros[block_idx]
            
            # Quantize and dequantize
            q_block = torch.clamp(torch.round(block / scale + zero), self.a_qmin, self.a_qmax)
            dq_block = (q_block - zero) * scale
            
            # Store back
            quantized_x[:, start_idx:end_idx] = dq_block
        
        return quantized_x
    
    def forward(self, x):
        # Apply activation quantization if needed
        if self.mode == 'calibration':
            x = self.calibrate_activation(x)
        elif self.mode == 'fake_quant':
            x = self.quantize_activation(x)
        
        # Apply weight quantization if needed
        if self.mode == 'float':
            return F.linear(x, self.weight, self.bias)
        else:  # fake_quant or calibration
            weight = self.quantize_weight()
            return F.linear(x, weight, self.bias)
    
    def set_mode(self, mode):
        """Set quantization mode."""
        assert mode in ['float', 'fake_quant', 'calibration']
        self.mode = mode


class PerBlockQuantizer:
    """Class to manage quantization of models."""
    def __init__(self, block_size=4, w_bits=8, a_bits=8):
        self.block_size = block_size
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.quantized_layers = []
        self.mode = 'float'
    
    def quantize_model(self, model):
        """Replace linear layers with quantized versions."""
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                # Create quantized layer
                quantized_layer = LinearPerBlockQuant(
                    module, 
                    block_size=self.block_size,
                    w_bits=self.w_bits,
                    a_bits=self.a_bits
                )
                # Replace original layer
                setattr(model, name, quantized_layer)
                self.quantized_layers.append(quantized_layer)
            else:
                # Recursively process child modules
                self.quantize_model(module)
        
        return model
    
    def set_mode(self, mode):
        """Set mode for all quantized layers."""
        self.mode = mode
        for layer in self.quantized_layers:
            layer.set_mode(mode)
    
    def calibrate_model(self, model, sample_inputs):
        """Calibrate the quantized model with sample inputs."""
        # Set calibration mode
        prev_mode = self.mode
        self.set_mode('calibration')
        
        # Forward pass to collect statistics
        with torch.no_grad():
            _ = model(sample_inputs)
        
        # Calculate quantization parameters for weights
        for layer in self.quantized_layers:
            layer.calibrate_weight()
        
        # Restore previous mode
        self.set_mode(prev_mode)
        
        return model

def get_model_input():
    torch.manual_seed(1234)
    model = SimpleModel(input_dim=8, hidden_dim=32, output_dim=8)
    batch_size = 10
    inputs = torch.randn(batch_size, 8)
    return model, inputs

def test_quantization_accuracy():
    """Test the accuracy difference between float and quantized models."""
    # Create model and sample data
    model, inputs = get_model_input()
    
    # Get reference outputs from float model
    model.eval()
    with torch.no_grad():
        float_outputs = model(inputs)
    
    # Test w8a8 quantization
    print("Testing w8a8-perblock quantization...")
    quantizer_w8a8 = PerBlockQuantizer(block_size=4, w_bits=8, a_bits=8)
    model_w8a8 = quantizer_w8a8.quantize_model(deepcopy(model))
    
    # Calibrate the model
    quantizer_w8a8.calibrate_model(model_w8a8, inputs)
    
    # Test in fake quant mode
    quantizer_w8a8.set_mode('fake_quant')
    with torch.no_grad():
        quant_outputs_w8a8 = model_w8a8(inputs)
    
    # Calculate MSE
    mse_w8a8 = F.mse_loss(float_outputs, quant_outputs_w8a8)
    print(f"MSE for w8a8-perblock: {mse_w8a8.item():.8f}")
    
    # Test w8a16 quantization
    print("\nTesting w8a16-perblock quantization...")
    quantizer_w8a16 = PerBlockQuantizer(block_size=4, w_bits=8, a_bits=16)
    model_w8a16 = quantizer_w8a16.quantize_model(deepcopy(model))
    
    # Calibrate the model
    quantizer_w8a16.calibrate_model(model_w8a16, inputs)
    
    # Test in fake quant mode
    quantizer_w8a16.set_mode('fake_quant')
    with torch.no_grad():
        quant_outputs_w8a16 = model_w8a16(inputs)
    
    # Calculate MSE
    mse_w8a16 = F.mse_loss(float_outputs, quant_outputs_w8a16)
    print(f"MSE for w8a16-perblock: {mse_w8a16.item():.8f}")
    
    return {
        'w8a8_mse': mse_w8a8.item(),
        'w8a16_mse': mse_w8a16.item()
    }


def test_different_block_sizes():
    """Test different block size configurations."""
    print("\nTesting different block sizes...")
    
    # Create model and sample data
    model, inputs = get_model_input()
    
    # Get reference outputs from float model
    model.eval()
    with torch.no_grad():
        float_outputs = model(inputs)
    
    # Test with tuple block_size
    print("Testing with block_size=(2, 4)...")
    quantizer_diff = PerBlockQuantizer(block_size=(2, 4), w_bits=8, a_bits=8)
    
    # Need to update the quantize_model method to pass the tuple block_size
    # This is a direct modification to handle tuple block_size
    def _modified_quantize_model(model):
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                # Create quantized layer with tuple block_size
                quantized_layer = LinearPerBlockQuant(
                    module, 
                    block_size=quantizer_diff.block_size,
                    w_bits=quantizer_diff.w_bits,
                    a_bits=quantizer_diff.a_bits
                )
                # Replace original layer
                setattr(model, name, quantized_layer)
                quantizer_diff.quantized_layers.append(quantized_layer)
            else:
                # Recursively process child modules
                _modified_quantize_model(module)
        return model
    
    model_diff = _modified_quantize_model(deepcopy(model))
    
    # Calibrate the model
    quantizer_diff.calibrate_model(model_diff, inputs)
    
    # Test in fake quant mode
    quantizer_diff.set_mode('fake_quant')
    with torch.no_grad():
        quant_outputs_diff = model_diff(inputs)
    
    # Calculate MSE
    mse_diff = F.mse_loss(float_outputs, quant_outputs_diff)
    print(f"MSE for different block sizes: {mse_diff.item():.8f}")
    
    return {
        'different_block_sizes_mse': mse_diff.item()
    }


def test_different_block_sizes():
    """Test the accuracy difference between float and quantized models."""
    print('\n\n', '-' * 50)
    
    # Create model and sample data
    model, inputs = get_model_input()
    
    # Get reference outputs from float model
    model.eval()
    with torch.no_grad():
        float_outputs = model(inputs)
    
    # Test w8a8 quantization
    block_size = (2, 4)
    quantizer_w8a8 = PerBlockQuantizer(block_size=block_size, w_bits=8, a_bits=8)
    print(f"Testing w8a8-perblock quantization...  block_size={block_size}")
    
    model_w8a8 = quantizer_w8a8.quantize_model(deepcopy(model))
    
    # Calibrate the model
    quantizer_w8a8.calibrate_model(model_w8a8, inputs)
    
    # Test in fake quant mode
    quantizer_w8a8.set_mode('fake_quant')
    with torch.no_grad():
        quant_outputs_w8a8 = model_w8a8(inputs)
    
    # Calculate MSE
    mse_w8a8 = F.mse_loss(float_outputs, quant_outputs_w8a8)
    print(f"MSE for w8a8-perblock: {mse_w8a8.item():.8f}")
    
    # Test w8a16 quantization
    print("\nTesting w8a16-perblock quantization...")
    quantizer_w8a16 = PerBlockQuantizer(block_size=4, w_bits=8, a_bits=16)
    model_w8a16 = quantizer_w8a16.quantize_model(deepcopy(model))
    
    # Calibrate the model
    quantizer_w8a16.calibrate_model(model_w8a16, inputs)
    
    # Test in fake quant mode
    quantizer_w8a16.set_mode('fake_quant')
    with torch.no_grad():
        quant_outputs_w8a16 = model_w8a16(inputs)
    
    # Calculate MSE
    mse_w8a16 = F.mse_loss(float_outputs, quant_outputs_w8a16)
    print(f"MSE for w8a16-perblock: {mse_w8a16.item():.8f}")
    
    return {
        'w8a8_mse': mse_w8a8.item(),
        'w8a16_mse': mse_w8a16.item()
    }

if __name__ == "__main__":
    results = test_quantization_accuracy()
    diff_results = test_different_block_sizes()
    print("\nTest completed successfully!")
