"""
Multi-Head Attention Implementation
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Args:
        d_model (int): Dimension of the model (embedding size)
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            K: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            V: Value tensor of shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor
            
        Returns:
            attention_output: Weighted values
            attention_weights: Attention weights
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights


if __name__ == "__main__":
    # Example usage
    print("Multi-Head Attention Example\n" + "="*50)
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass (self-attention: Q, K, V are all the same)
    output, attention_weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nNumber of parameters: {sum(p.numel() for p in mha.parameters()):,}")
    print("\n✓ Multi-Head Attention implementation successful!")
