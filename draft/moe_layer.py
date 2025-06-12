import tensorflow as tf
from tensorflow.keras import layers

from config import NUM_EXPERTS, EXPERT_CAPACITY_FACTOR, EXPERT_CAPACITY_ADAPTATION, MIN_EXPERT_CAPACITY, MAX_EXPERT_CAPACITY, ADAPTATION_INTERVAL, HIERARCHICAL_MOE, NUM_LEVELS, EXPERTS_PER_LEVEL, SPECIALIZATION_REGULARIZATION, SPECIALIZATION_LOSS_WEIGHT
from router import MoERouter

class MoELayer(layers.Layer):
    def __init__(self, d_model, num_experts=NUM_EXPERTS, k=2, dff=2048, dropout_rate=0.1):
        super(MoELayer, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.dff = dff
        self.capacity_factor = EXPERT_CAPACITY_FACTOR
        self.adapt_capacity = EXPERT_CAPACITY_ADAPTATION
        self.min_capacity = MIN_EXPERT_CAPACITY
        self.max_capacity = MAX_EXPERT_CAPACITY
        self.adaptation_interval = ADAPTATION_INTERVAL
        self.hierarchical_moe = HIERARCHICAL_MOE
        self.num_levels = NUM_LEVELS if HIERARCHICAL_MOE else 1
        self.experts_per_level = EXPERTS_PER_LEVEL if HIERARCHICAL_MOE else [num_experts]
        self.specialization_regularization = SPECIALIZATION_REGULARIZATION
        
        # Router for expert assignment (one per level if hierarchical)
        self.routers = []
        self.expert_groups = []
        self.expert_capacities = []
        total_experts = sum(self.experts_per_level) if HIERARCHICAL_MOE else num_experts
        
        if self.hierarchical_moe:
            for level in range(self.num_levels):
                router = MoERouter(d_model, self.experts_per_level[level], k)
                self.routers.append(router)
                level_experts = []
                level_capacities = [1.0] * self.experts_per_level[level]
                for _ in range(self.experts_per_level[level]):
                    expert = tf.keras.Sequential([
                        layers.Dense(dff, activation='relu'),
                        layers.Dropout(dropout_rate),
                        layers.Dense(d_model)
                    ])
                    level_experts.append(expert)
                self.expert_groups.append(level_experts)
                self.expert_capacities.append(level_capacities)
        else:
            router = MoERouter(d_model, num_experts, k)
            self.routers.append(router)
            level_experts = []
            level_capacities = [1.0] * num_experts
            for _ in range(num_experts):
                expert = tf.keras.Sequential([
                    layers.Dense(dff, activation='relu'),
                    layers.Dropout(dropout_rate),
                    layers.Dense(d_model)
                ])
                level_experts.append(expert)
            self.expert_groups.append(level_experts)
            self.expert_capacities.append(level_capacities)
        
        self.dropout = layers.Dropout(dropout_rate)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.expert_usage_history = [tf.Variable(tf.zeros(num_exp), trainable=False) for num_exp in self.experts_per_level] if HIERARCHICAL_MOE else [tf.Variable(tf.zeros(num_experts), trainable=False)]
        
    def adapt_expert_capacities(self, router_probs_list):
        """Adapt the capacity of experts based on their usage"""
        if not self.adapt_capacity:
            return
        
        self.global_step.assign_add(1)
        for level in range(self.num_levels):
            router_probs = router_probs_list[level] if self.hierarchical_moe else router_probs_list[0]
            batch_usage = tf.reduce_mean(tf.reduce_sum(router_probs, axis=1), axis=0)  # Shape: (num_experts_level,)
            self.expert_usage_history[level].assign_add(batch_usage)
        
        if tf.math.mod(self.global_step, self.adaptation_interval) == 0:
            for level in range(self.num_levels):
                usage_mean = self.expert_usage_history[level] / tf.cast(self.adaptation_interval, tf.float32)
                for i in range(len(self.expert_groups[level])):
                    new_capacity = self.expert_capacities[level][i]
                    if usage_mean[i] > 0.9:  # High usage
                        new_capacity = min(new_capacity + 0.1, self.max_capacity)
                    elif usage_mean[i] < 0.1:  # Low usage
                        new_capacity = max(new_capacity - 0.1, self.min_capacity)
                    self.expert_capacities[level][i] = new_capacity
                    # Update expert architecture if necessary (simplified here as capacity factor)
                    if hasattr(self.expert_groups[level][i].layers[0], 'units'):
                        new_dff = int(self.dff * new_capacity)
                        self.expert_groups[level][i].layers[0] = layers.Dense(new_dff, activation='relu')
                        self.expert_groups[level][i].layers[2] = layers.Dense(self.d_model)
                self.expert_usage_history[level].assign(tf.zeros(len(self.expert_groups[level])))
    
    def compute_specialization_loss(self):
        """Compute loss to encourage specialization among experts"""
        if not self.specialization_regularization:
            return 0.0
        
        specialization_loss = 0.0
        for level in range(self.num_levels):
            # Get weights of the first dense layer of each expert at this level
            expert_weights = [expert.layers[0].get_weights()[0] for expert in self.expert_groups[level]]
            num_experts = len(expert_weights)
            if num_experts < 2:
                continue
            
            # Compute pairwise differences between expert weights
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    diff = tf.reduce_mean(tf.square(expert_weights[i] - expert_weights[j]))
                    specialization_loss += diff
        
        # Normalize by number of levels and return inverse (maximize difference)
        specialization_loss = -specialization_loss / self.num_levels
        return specialization_loss * SPECIALIZATION_LOSS_WEIGHT
    
    def call(self, inputs, training=True):
        batch_size, seq_len, _ = tf.shape(inputs)
        current_inputs = inputs
        router_probs_list = []
        routing_outputs_list = []
        
        # Process through hierarchy of experts
        for level in range(self.num_levels):
            # Get routing decisions for this level
            routing_outputs = self.routers[level](current_inputs, training=training)
            dispatch_tensor = routing_outputs['dispatch_tensor']
            router_probs = routing_outputs['router_probs']
            top_k_probs = routing_outputs['top_k_probs']
            
            router_probs_list.append(router_probs)
            routing_outputs_list.append(routing_outputs)
            
            # Initialize output tensor for this level
            level_outputs = tf.zeros_like(current_inputs)
            
            # Process each expert at this level
            for expert_idx in range(len(self.expert_groups[level])):
                expert_mask = dispatch_tensor[:, :, expert_idx:expert_idx+1]
                expert_mask = tf.tile(expert_mask, [1, 1, self.d_model])
                expert_inputs = current_inputs * expert_mask
                expert_outputs = self.expert_groups[level][expert_idx](expert_inputs, training=training)
                expert_prob = router_probs[:, :, expert_idx:expert_idx+1]
                expert_prob = tf.tile(expert_prob, [1, 1, self.d_model])
                weighted_expert_outputs = expert_outputs * expert_prob
                level_outputs += weighted_expert_outputs
            
            current_inputs = level_outputs  # Output of this level becomes input to next level
        
        outputs = current_inputs  # Final output after all levels
        
        if training:
            self.adapt_expert_capacities(router_probs_list)
            outputs = self.dropout(outputs, training=training)
        
        return outputs, routing_outputs_list
    
    def get_load_balance_loss(self, router_probs_list):
        total_loss = 0.0
        for level in range(self.num_levels):
            total_loss += self.routers[level].compute_load_balance_loss(router_probs_list[level])
        return total_loss / self.num_levels
    
    def get_specialization_loss(self):
        return self.compute_specialization_loss() 