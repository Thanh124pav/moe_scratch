import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from config import NUM_EXPERTS, EXPERTS_PER_TOKEN, ROUTER_TYPE, CONTEXT_AWARE_ROUTING, CONTEXT_WINDOW_SIZE

class MoERouter(layers.Layer):
    def __init__(self, d_model, num_experts=NUM_EXPERTS, k=EXPERTS_PER_TOKEN, router_type=ROUTER_TYPE):
        super(MoERouter, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.router_type = router_type
        self.context_aware = CONTEXT_AWARE_ROUTING
        self.context_window = CONTEXT_WINDOW_SIZE if CONTEXT_AWARE_ROUTING else 1
        
        # Router dense layer to compute expert assignment logits
        self.router_dense = layers.Dense(num_experts, use_bias=False)
        
        if self.context_aware:
            # Small attention layer for context awareness
            self.context_attention = layers.MultiHeadAttention(num_heads=2, key_dim=d_model//2)
        
    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        """Sample from the Gumbel-Softmax distribution"""
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits))))
        y = logits + gumbel_noise
        y = tf.nn.softmax(y / temperature)
        
        if hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
    
    def call(self, inputs, training=True):
        batch_size, seq_len, _ = tf.shape(inputs)
        
        if self.context_aware:
            # Apply context attention to consider neighboring tokens
            context_inputs = inputs
            context_output = self.context_attention(context_inputs, context_inputs, context_inputs)
            inputs = inputs + context_output
        
        # Compute router logits
        router_logits = self.router_dense(inputs)  # Shape: (batch_size, seq_len, num_experts)
        
        if self.router_type == 'gumbel':
            router_probs = self.gumbel_softmax(router_logits, temperature=1.0, hard=False)
        else:
            router_probs = tf.nn.softmax(router_logits)
        
        # Select top-k experts
        top_k_probs, top_k_indices = tf.nn.top_k(router_probs, k=self.k, sorted=False)
        
        # Create dispatch tensor (one-hot encoding of selected experts)
        dispatch_tensor = tf.zeros((batch_size, seq_len, self.num_experts))
        updates = tf.ones_like(top_k_indices, dtype=tf.float32)
        dispatch_tensor = tf.tensor_scatter_nd_add(
            dispatch_tensor,
            tf.stack([tf.tile(tf.range(batch_size)[:, tf.newaxis, tf.newaxis], [1, seq_len, self.k]),
                      tf.tile(tf.range(seq_len)[tf.newaxis, :, tf.newaxis], [batch_size, 1, self.k]),
                      top_k_indices], axis=-1),
            updates
        )
        
        return {
            'dispatch_tensor': dispatch_tensor,  # Binary matrix indicating selected experts
            'router_probs': router_probs,       # Full probability distribution over experts
            'top_k_probs': top_k_probs,         # Probabilities of selected experts
            'top_k_indices': top_k_indices      # Indices of selected experts
        }
    
    def compute_load_balance_loss(self, router_probs):
        """Compute auxiliary loss to encourage load balancing across experts"""
        batch_size = tf.shape(router_probs)[0]
        expert_usage = tf.reduce_mean(tf.reduce_sum(router_probs, axis=1), axis=0)  # Shape: (num_experts,)
        expert_usage_mean = tf.reduce_mean(expert_usage)
        expert_usage_var = tf.reduce_mean(tf.square(expert_usage - expert_usage_mean))
        return expert_usage_var 