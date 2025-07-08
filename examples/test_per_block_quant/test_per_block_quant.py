import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, original_layer, block_size=4, w_bits=8, a_bits=8):
        super(LinearPerBlockQuant, self).__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.block_size = block_size
        
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
        self.num_blocks_in_fea = (self.in_features + block_size - 1) // block_size
        self.num_blocks_out_fea = (self.out_features + block_size - 1) // block_size
              
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
                start_idx = block_idx * self.block_size
                end_idx = min(start_idx + self.block_size, self.in_features)
                
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
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, self.in_features)
            
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
                start_idx = block_idx * self.block_size
                end_idx = min(start_idx + self.block_size, self.in_features)
                
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
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, self.in_features)
            
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


from copy import deepcopy
def test_quantization_accuracy():
    """Test the accuracy difference between float and quantized models."""
    # Create model and sample data
    model = SimpleModel(input_dim=8, hidden_dim=32, output_dim=8)
    batch_size = 10
    inputs = torch.randn(batch_size, 8)
    
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


if __name__ == "__main__":
    results = test_quantization_accuracy()
    print("\nTest completed successfully!")
