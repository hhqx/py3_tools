#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test MultiheadAttention Replacement System

This script implements and tests a replacement system for PyTorch's nn.MultiheadAttention.
It defines a custom EagerMultiheadAttention implementation and provides utilities
to replace nn.MultiheadAttention instances in models with this custom implementation.

The file includes:
1. A SimpleModel using PyTorch's MultiheadAttention
2. A custom EagerMultiheadAttention implementation
3. A replacement function to swap MultiheadAttention instances
4. Test functions to verify correctness of the implementation and replacement

Usage:
    python test_mha_attn.py [--debug]
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s][%(name)s][%(funcName)s] %(message)s'
)
logger = logging.getLogger("mha_test")


class SimpleModel(nn.Module):
    """
    A simple model using PyTorch's MultiheadAttention for testing purposes.
    
    Attributes:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mha (nn.MultiheadAttention): PyTorch's MultiheadAttention module
    """
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.0):
        """
        Initialize the SimpleModel.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(SimpleModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Forward pass through the model.
        
        Args:
            query: Query tensor of shape (seq_len, batch_size, embed_dim)
            key: Key tensor of shape (seq_len, batch_size, embed_dim)
            value: Value tensor of shape (seq_len, batch_size, embed_dim)
            key_padding_mask: Mask for keys (optional)
            attn_mask: Attention mask (optional)
            
        Returns:
            tuple: (output, attention_weights)
        """
        output, attention_weights = self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        return output, attention_weights


class EagerMultiheadAttention(nn.Module):
    """
    Custom implementation of MultiheadAttention using eager execution
    with the same interface as torch.nn.MultiheadAttention.
    
    Implements multi-head attention as described in "Attention Is All You Need".
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, 
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None):
        """
        Initialize the EagerMultiheadAttention module.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: If specified, adds bias to input linear transformations
            add_bias_kv: Add bias to the key and value sequences at dim=0
            add_zero_attn: Add a new batch of zeros to the key and value sequences at dim=1
            kdim: Total number of features for keys (default: embed_dim)
            vdim: Total number of features for values (default: embed_dim)
            batch_first: If True, input and output tensors are provided as 
                         (batch, seq, feature) instead of (seq, batch, feature)
            device: Device to initialize parameters on
            dtype: Data type of parameters
        """
        super(EagerMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Optional parameters
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        
        if self.add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None
            
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize or reset all the parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
            
        if self.add_bias_kv:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)
    
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query embeddings (seq_len, batch_size, embed_dim) or 
                  (batch_size, seq_len, embed_dim) if batch_first=True
            key: Key embeddings
            value: Value embeddings
            key_padding_mask: Mask to exclude keys that are just padding
            need_weights: If True, returns attention weights
            attn_mask: 2D or 3D mask preventing attention to certain positions
            average_attn_weights: If True, returns averaged attention weights per head
            is_causal: If True, applies a causal mask
            
        Returns:
            tuple: (attention output, attention weights)
        """
        # Handle batch_first format by transposing
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Extract dimensions
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Compute scaled dot-product attention
        scaling = float(self.head_dim) ** -0.5
        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) * scaling
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask
            
        if is_causal:
            # Create a causal mask
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device),
                diagonal=1
            )
            attn_output_weights.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
            
        # Handle key padding mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask, float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Apply softmax and dropout
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Get the weighted sum
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        # Handle attention weights for return value
        if need_weights:
            # Reshape attention weights
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            attn_output_weights = None
        
        # Restore batch_first format if needed
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
            
        return attn_output, attn_output_weights


class EagerMultiheadAttentionWithInProj(nn.Module):
    """
    Custom implementation of MultiheadAttention using eager execution with shared in_proj
    weights for Q, K, V projections, mimicking the original PyTorch implementation more closely.
    
    This implementation uses a single Linear layer for projecting query, key and value,
    which is identical to how PyTorch's nn.MultiheadAttention handles projections internally.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, 
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None):
        """
        Initialize the EagerMultiheadAttentionWithInProj module.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: If specified, adds bias to input linear transformations
            add_bias_kv: Add bias to the key and value sequences at dim=0
            add_zero_attn: Add a new batch of zeros to the key and value sequences at dim=1
            kdim: Total number of features for keys (default: embed_dim)
            vdim: Total number of features for values (default: embed_dim)
            batch_first: If True, input and output tensors are provided as 
                         (batch, seq, feature) instead of (seq, batch, feature)
            device: Device to initialize parameters on
            dtype: Data type of parameters
        """
        super(EagerMultiheadAttentionWithInProj, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Single shared projection for q, k, v (similar to PyTorch implementation)
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
            
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Optional parameters
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        
        if self.add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None
            
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize or reset all the parameters."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
            
        if self.add_bias_kv:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)
    
    def _in_projection(self, query, key, value):
        """
        Performs the in-projection step of the attention operation.
        This is simply a triple of linear projections.
        
        Args:
            query, key, value: Input tensors
            
        Returns:
            q, k, v: Projected tensors
        """
        w = self.in_proj_weight
        b = self.in_proj_bias
        
        # Split weights for q, k, v projections
        w_q, w_k, w_v = w.chunk(3)
        if b is not None:
            b_q, b_k, b_v = b.chunk(3)
        else:
            b_q = b_k = b_v = None
            
        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)
        
        return q, k, v
            
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        """
        Forward pass of multi-head attention with shared projection weights.
        
        Args:
            query: Query embeddings (seq_len, batch_size, embed_dim) or 
                  (batch_size, seq_len, embed_dim) if batch_first=True
            key: Key embeddings
            value: Value embeddings
            key_padding_mask: Mask to exclude keys that are just padding
            need_weights: If True, returns attention weights
            attn_mask: 2D or 3D mask preventing attention to certain positions
            average_attn_weights: If True, returns averaged attention weights per head
            is_causal: If True, applies a causal mask
            
        Returns:
            tuple: (attention output, attention weights)
        """
        # Handle batch_first format by transposing
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Extract dimensions
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        # Apply linear projections using the shared weights
        q, k, v = self._in_projection(query, key, value)
        
        # Reshape for multi-head attention
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Compute scaled dot-product attention
        scaling = float(self.head_dim) ** -0.5
        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) * scaling
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask
            
        if is_causal:
            # Create a causal mask
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device),
                diagonal=1
            )
            attn_output_weights.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
            
        # Handle key padding mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask, float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Apply softmax and dropout
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Get the weighted sum
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        # Handle attention weights for return value
        if need_weights:
            # Reshape attention weights
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            attn_output_weights = None
        
        # Restore batch_first format if needed
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
            
        return attn_output, attn_output_weights


def replace_mha_with_eager(model, use_in_proj=False):
    """
    Replace all instances of nn.MultiheadAttention with EagerMultiheadAttention
    in a model, preserving weights.
    
    Args:
        model: PyTorch model to modify
        use_in_proj: If True, uses EagerMultiheadAttentionWithInProj instead of
                    EagerMultiheadAttention (default: False)
        
    Returns:
        model: Modified PyTorch model
    """
    logger.info(f"Starting replacement of MultiheadAttention modules with {'in_proj version' if use_in_proj else 'separate qkv version'}")
    
    # Create a deep copy to avoid modifying the original model
    model = copy.deepcopy(model)
    
    # Track replacements
    replacement_count = 0
    
    # Select which implementation to use
    eager_class = EagerMultiheadAttentionWithInProj if use_in_proj else EagerMultiheadAttention
    
    # Helper function to recursively replace modules
    def replace_modules(module, parent_name=""):
        nonlocal replacement_count
        
        # Store modifications to make after iteration
        replacements = {}
        
        # Iterate through all named children
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # If the child is a MultiheadAttention, replace it
            if isinstance(child, nn.MultiheadAttention):
                # Extract parameters from original MHA
                embed_dim = child.embed_dim
                num_heads = child.num_heads
                dropout = child.dropout
                bias = child.in_proj_bias is not None
                add_bias_kv = child.bias_k is not None
                batch_first = child.batch_first
                
                # Create new EagerMultiheadAttention with same parameters
                new_mha = eager_class(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                    add_bias_kv=add_bias_kv,
                    batch_first=batch_first
                )
                
                # Transfer weights from PyTorch MHA to our implementation
                with torch.no_grad():
                    if use_in_proj:
                        # For in_proj version
                        if child.in_proj_weight is not None:
                            new_mha.in_proj_weight.copy_(child.in_proj_weight)
                        
                        if child.in_proj_bias is not None:
                            new_mha.in_proj_bias.copy_(child.in_proj_bias)
                    else:
                        # For separate qkv projections version
                        if child.in_proj_weight is not None:
                            # PyTorch concatenates q,k,v projections
                            q_weight, k_weight, v_weight = child.in_proj_weight.chunk(3, dim=0)
                            new_mha.q_proj.weight.copy_(q_weight)
                            new_mha.k_proj.weight.copy_(k_weight)
                            new_mha.v_proj.weight.copy_(v_weight)
                        else:
                            # Handle separate q,k,v projections
                            if child.q_proj_weight is not None:
                                new_mha.q_proj.weight.copy_(child.q_proj_weight)
                            if child.k_proj_weight is not None:
                                new_mha.k_proj.weight.copy_(child.k_proj_weight)
                            if child.v_proj_weight is not None:
                                new_mha.v_proj.weight.copy_(child.v_proj_weight)
                        
                        # Handle biases for separate projections
                        if child.in_proj_bias is not None:
                            q_bias, k_bias, v_bias = child.in_proj_bias.chunk(3, dim=0)
                            new_mha.q_proj.bias.copy_(q_bias)
                            new_mha.k_proj.bias.copy_(k_bias)
                            new_mha.v_proj.bias.copy_(v_bias)
                    
                    # Output projection (common for both implementations)
                    new_mha.out_proj.weight.copy_(child.out_proj.weight)
                    if child.out_proj.bias is not None:
                        new_mha.out_proj.bias.copy_(child.out_proj.bias)
                    
                    # Copy bias_k, bias_v if they exist
                    if child.bias_k is not None and new_mha.bias_k is not None:
                        new_mha.bias_k.copy_(child.bias_k)
                        new_mha.bias_v.copy_(child.bias_v)
                
                # Save for replacement
                replacements[name] = new_mha
                replacement_count += 1
                logger.debug(f"Scheduled replacement for {full_name}")
            else:
                # Recursively process children
                replace_modules(child, full_name)
        
        # Apply replacements
        for name, new_module in replacements.items():
            module._modules[name] = new_module
    
    # Start the recursive replacement process
    replace_modules(model)
    
    logger.info(f"Replaced {replacement_count} MultiheadAttention modules")
    return model


def test_model_forward():
    """
    Test the forward pass of the SimpleModel to ensure it works correctly.
    """
    logger.info("Testing SimpleModel forward pass")
    
    # Create a model instance
    model = SimpleModel(embed_dim=64, num_heads=4)
    
    # Create dummy inputs
    seq_len, batch_size, embed_dim = 10, 2, 64
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)
    
    # Test forward pass
    try:
        output, attn_weights = model(query, key, value)
        
        # Check output shapes
        assert output.shape == (seq_len, batch_size, embed_dim), f"Expected output shape {(seq_len, batch_size, embed_dim)}, got {output.shape}"
        assert attn_weights.shape == (batch_size, seq_len, seq_len), f"Expected attention weights shape {(batch_size, seq_len, seq_len)}, got {attn_weights.shape}"
        
        # Check output values are not NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        assert torch.isfinite(attn_weights).all(), "Attention weights contain NaN or Inf values"
        
        logger.info("SimpleModel forward pass test passed ✓")
        return True
    except Exception as e:
        logger.error(f"SimpleModel forward pass test failed: {e}")
        return False


def test_mha_weight_consistency():
    """
    Test that weights are consistently transferred during replacement.
    """
    logger.info("Testing weight consistency after MHA replacement")
    
    # Create a model instance
    original_model = SimpleModel(embed_dim=64, num_heads=4)
    
    # Replace MultiheadAttention with EagerMultiheadAttention
    replaced_model = replace_mha_with_eager(original_model)
    
    # Check that original model still has nn.MultiheadAttention
    assert isinstance(original_model.mha, nn.MultiheadAttention), "Original model's MHA was modified"
    
    # Check that replaced model has EagerMultiheadAttention
    assert isinstance(replaced_model.mha, EagerMultiheadAttention), "MHA was not replaced correctly"
    
    # Compare weights
    try:
        # Get original weights
        orig_mha = original_model.mha
        replaced_mha = replaced_model.mha
        
        # Check Q projection
        q_weight_orig = orig_mha.in_proj_weight[:orig_mha.embed_dim]
        q_weight_replaced = replaced_mha.q_proj.weight
        torch.testing.assert_close(q_weight_orig, q_weight_replaced, 
                                  msg="Q projection weights differ")
        
        # Check K projection
        k_weight_orig = orig_mha.in_proj_weight[orig_mha.embed_dim:2*orig_mha.embed_dim]
        k_weight_replaced = replaced_mha.k_proj.weight
        torch.testing.assert_close(k_weight_orig, k_weight_replaced, 
                                  msg="K projection weights differ")
        
        # Check V projection
        v_weight_orig = orig_mha.in_proj_weight[2*orig_mha.embed_dim:]
        v_weight_replaced = replaced_mha.v_proj.weight
        torch.testing.assert_close(v_weight_orig, v_weight_replaced, 
                                  msg="V projection weights differ")
        
        # Check output projection
        torch.testing.assert_close(orig_mha.out_proj.weight, replaced_mha.out_proj.weight,
                                  msg="Output projection weights differ")
        
        # Check biases
        if orig_mha.in_proj_bias is not None:
            q_bias_orig = orig_mha.in_proj_bias[:orig_mha.embed_dim]
            k_bias_orig = orig_mha.in_proj_bias[orig_mha.embed_dim:2*orig_mha.embed_dim]
            v_bias_orig = orig_mha.in_proj_bias[2*orig_mha.embed_dim:]
            
            torch.testing.assert_close(q_bias_orig, replaced_mha.q_proj.bias,
                                      msg="Q projection bias differs")
            torch.testing.assert_close(k_bias_orig, replaced_mha.k_proj.bias,
                                      msg="K projection bias differs")
            torch.testing.assert_close(v_bias_orig, replaced_mha.v_proj.bias,
                                      msg="V projection bias differs")
        
        if orig_mha.out_proj.bias is not None:
            torch.testing.assert_close(orig_mha.out_proj.bias, replaced_mha.out_proj.bias,
                                      msg="Output projection bias differs")
        
        logger.info("MHA weight consistency test passed ✓")
        return True
    except Exception as e:
        logger.error(f"MHA weight consistency test failed: {e}")
        return False


def test_model_output_consistency():
    """
    Test that model outputs are consistent before and after replacement.
    """
    logger.info("Testing model output consistency")
    
    # Create random inputs
    seq_len, batch_size, embed_dim = 8, 4, 64
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)
    
    # Create attention mask
    attn_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1
    ).to(torch.float) * -1e9
    
    # Create a key padding mask
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    key_padding_mask[:, -2:] = True  # Mask the last two positions
    
    # Create a model instance with fixed seed for reproducibility
    torch.manual_seed(42)
    original_model = SimpleModel(embed_dim=embed_dim, num_heads=4, dropout=0.0)
    original_model.eval()  # Set to evaluation mode to disable dropout
    
    # Replace MultiheadAttention with EagerMultiheadAttention
    replaced_model = replace_mha_with_eager(original_model)
    replaced_model.eval()  # Set to evaluation mode to disable dropout
    
    try:
        # Get outputs from original model
        with torch.no_grad():
            orig_output, orig_attn_weights = original_model(
                query, key, value, 
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        
        # Get outputs from replaced model
        with torch.no_grad():
            replaced_output, replaced_attn_weights = replaced_model(
                query, key, value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        
        # Compare outputs
        torch.testing.assert_close(
            orig_output, replaced_output,
            rtol=1e-4, atol=1e-4,
            msg="Model outputs differ after replacement"
        )
        
        # Compare attention weights
        torch.testing.assert_close(
            orig_attn_weights, replaced_attn_weights,
            rtol=1e-4, atol=1e-4,
            msg="Attention weights differ after replacement"
        )
        
        logger.info("Model output consistency test passed ✓")
        return True
    except Exception as e:
        logger.error(f"Model output consistency test failed: {e}")
        return False


def test_replace_mha():
    """
    Test the replace_mha_with_eager function with a complex model.
    """
    logger.info("Testing MHA replacement in a complex model")
    
    # Create a more complex model with nested MHAs
    class ComplexModel(nn.Module):
        def __init__(self):
            super(ComplexModel, self).__init__()
            self.mha1 = nn.MultiheadAttention(embed_dim=32, num_heads=2)
            self.layer1 = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU()
            )
            self.nested = nn.ModuleDict({
                'mha2': nn.MultiheadAttention(embed_dim=64, num_heads=4),
                'block': nn.Sequential(
                    nn.Linear(64, 64),
                    nn.MultiheadAttention(embed_dim=64, num_heads=8)
                )
            })
        
        def forward(self, x):
            # This is just a placeholder, we don't need actual forward logic for this test
            return x
    
    # Create an instance
    original_model = ComplexModel()
    
    # Replace MHAs
    replaced_model = replace_mha_with_eager(original_model)
    
    try:
        # Check that all MHAs were replaced
        assert isinstance(replaced_model.mha1, EagerMultiheadAttention), "mha1 was not replaced"
        assert isinstance(replaced_model.nested['mha2'], EagerMultiheadAttention), "mha2 was not replaced"
        assert isinstance(replaced_model.nested['block'][1], EagerMultiheadAttention), "nested mha was not replaced"
        
        # Check that the original model is unchanged
        assert isinstance(original_model.mha1, nn.MultiheadAttention), "Original model was modified"
        assert isinstance(original_model.nested['mha2'], nn.MultiheadAttention), "Original model was modified"
        assert isinstance(original_model.nested['block'][1], nn.MultiheadAttention), "Original model was modified"
        
        logger.info("Complex model replacement test passed ✓")
        return True
    except Exception as e:
        logger.error(f"Complex model replacement test failed: {e}")
        return False


def test_in_proj_mha_weight_consistency():
    """
    Test that weights are consistently transferred during replacement with in_proj version.
    """
    logger.info("Testing weight consistency after MHA replacement with in_proj version")
    
    # Create a model instance
    original_model = SimpleModel(embed_dim=64, num_heads=4)
    
    # Replace MultiheadAttention with EagerMultiheadAttentionWithInProj
    replaced_model = replace_mha_with_eager(original_model, use_in_proj=True)
    
    # Check that original model still has nn.MultiheadAttention
    assert isinstance(original_model.mha, nn.MultiheadAttention), "Original model's MHA was modified"
    
    # Check that replaced model has EagerMultiheadAttentionWithInProj
    assert isinstance(replaced_model.mha, EagerMultiheadAttentionWithInProj), "MHA was not replaced correctly"
    
    # Compare weights
    try:
        # Get original weights
        orig_mha = original_model.mha
        replaced_mha = replaced_model.mha
        
        # Check in_proj weight
        torch.testing.assert_close(
            orig_mha.in_proj_weight, replaced_mha.in_proj_weight,
            msg="in_proj weights differ"
        )
        
        # Check in_proj bias if exists
        if orig_mha.in_proj_bias is not None:
            torch.testing.assert_close(
                orig_mha.in_proj_bias, replaced_mha.in_proj_bias,
                msg="in_proj bias differs"
            )
        
        # Check output projection
        torch.testing.assert_close(
            orig_mha.out_proj.weight, replaced_mha.out_proj.weight,
            msg="Output projection weights differ"
        )
        
        if orig_mha.out_proj.bias is not None:
            torch.testing.assert_close(
                orig_mha.out_proj.bias, replaced_mha.out_proj.bias,
                msg="Output projection bias differs"
            )
        
        logger.info("In-proj MHA weight consistency test passed ✓")
        return True
    except Exception as e:
        logger.error(f"In-proj MHA weight consistency test failed: {e}")
        return False


def test_in_proj_model_output_consistency():
    """
    Test that model outputs are consistent before and after replacement with in_proj version.
    """
    logger.info("Testing model output consistency with in_proj version")
    
    # Create random inputs
    seq_len, batch_size, embed_dim = 8, 4, 64
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)
    
    # Create attention mask
    attn_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1
    ).to(torch.float) * -1e9
    
    # Create a key padding mask
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    key_padding_mask[:, -2:] = True  # Mask the last two positions
    
    # Create a model instance with fixed seed for reproducibility
    torch.manual_seed(42)
    original_model = SimpleModel(embed_dim=embed_dim, num_heads=4, dropout=0.0)
    original_model.eval()  # Set to evaluation mode to disable dropout
    
    # Replace MultiheadAttention with EagerMultiheadAttentionWithInProj
    replaced_model = replace_mha_with_eager(original_model, use_in_proj=True)
    replaced_model.eval()  # Set to evaluation mode to disable dropout
    
    try:
        # Get outputs from original model
        with torch.no_grad():
            orig_output, orig_attn_weights = original_model(
                query, key, value, 
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        
        # Get outputs from replaced model
        with torch.no_grad():
            replaced_output, replaced_attn_weights = replaced_model(
                query, key, value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask
            )
        
        # Compare outputs
        torch.testing.assert_close(
            orig_output, replaced_output,
            rtol=1e-4, atol=1e-4,
            msg="Model outputs differ after replacement with in_proj version"
        )
        
        # Compare attention weights
        torch.testing.assert_close(
            orig_attn_weights, replaced_attn_weights,
            rtol=1e-4, atol=1e-4,
            msg="Attention weights differ after replacement with in_proj version"
        )
        
        logger.info("Model output consistency test with in_proj version passed ✓")
        return True
    except Exception as e:
        logger.error(f"Model output consistency test with in_proj version failed: {e}")
        return False


def main():
    """
    Main function to run all tests.
    """
    parser = argparse.ArgumentParser(description='Test MultiheadAttention replacement')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--use-in-proj', action='store_true', help='Test with in_proj version')
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    logger.info("Starting MultiheadAttention replacement tests")
    
    # Run all tests
    tests = [
        test_model_forward,
        test_mha_weight_consistency,
        test_model_output_consistency,
        test_replace_mha,
        test_in_proj_mha_weight_consistency,
        test_in_proj_model_output_consistency
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    # Summarize results
    total = len(tests)
    passed = sum(results)
    logger.info(f"Test summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests passed! 🎉")
        return 0
    else:
        logger.error(f"Some tests failed ({total-passed}/{total})")
        return 1


if __name__ == "__main__":
    main()
