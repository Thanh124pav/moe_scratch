import torch.nn as nn
import torch
import torch.nn.functional as F
#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 2 * n_embed),
            nn.ReLU(),
            nn.Linear(2 * n_embed, n_embed),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output): 
        # mh_output: (B, T, C)
        logits = self.topkroute_linear(mh_output) # (B, T, n_experts)
        noise_logits = self.noise_linear(mh_output) # (B, T, n_experts)
        noise = torch.randn_like(logits) * F.softplus(noise_logits) # (B, T, n_experts)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        # top_k_logits: (B, T, top_k (probs) ) , indices: (B, T, top_k(indices))
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_probs = F.softmax(sparse_logits, dim=-1) # (B, T, n_experts)

        # Load balancing loss
        expert_prob = router_probs.mean(dim=(0, 1))  # (num_experts, 1) xác suất tb các experts được chọn 
        expert_mask = torch.zeros_like(router_probs).scatter(-1, indices, 1.0)
        expert_count = expert_mask.mean(dim=(0, 1)) # (num_experts, 1) số lần (trung bình) các experts được chọn
        load_balance_loss = self.num_experts * (expert_prob * expert_count).sum()

        return router_probs, indices, load_balance_loss


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.1):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        gating_output, indices, load_balance_loss = self.router(x)
        #gating_output: (B, T, n_experts)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1)) # (B*T, C)
        flat_gating_output = gating_output.view(-1, gating_output.size(-1)) # (B*T, n_experts)

        tokens_per_batch = batch_size * seq_len * self.top_k 
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1) # (B, T) 
            flat_mask = expert_mask.view(-1) # (B*T, 1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1) # (num_selected, )

            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices] # (selected_indices, C)
                expert_output = expert(expert_input) # (selected_indices, C)

                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1) # (selected_indices, 1)
                weighted_output = expert_output * gating_scores

                updates.index_add_(0, limited_indices, weighted_output)

        final_output += updates.view(batch_size, seq_len, -1)
        return final_output, load_balance_loss

# class SparseMoEWithShared(nn.Module):
#     def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.1):
#         super(SparseMoEWithShared, self).__init__()
#         # Routed experts setup
#         self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
#         self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
#         self.top_k = top_k
#         self.capacity_factor = capacity_factor
#         self.num_experts = num_experts
        
#         # Shared expert setup
#         self.shared_expert = Expert(n_embed)
#         # Weight for balancing routed vs shared outputs
#         self.routed_weight = nn.Parameter(torch.ones(1))
#         self.shared_weight = nn.Parameter(torch.ones(1))
    
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
        
#         # Process through routed experts
#         gating_output, indices, load_balance_loss = self.router(x)
#         final_output = torch.zeros_like(x)
        
#         flat_x = x.view(-1, x.size(-1))
#         flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        
#         tokens_per_batch = batch_size * seq_len * self.top_k
#         expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
#         updates = torch.zeros_like(flat_x)
        
#         # Process through routed experts
#         for i, expert in enumerate(self.experts):
#             expert_mask = (indices == i).any(dim=-1)
#             flat_mask = expert_mask.view(-1)
#             selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            
#             limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
#             if limited_indices.numel() > 0:
#                 expert_input = flat_x[limited_indices]
#                 expert_output = expert(expert_input)
                
#                 gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
#                 weighted_output = expert_output * gating_scores
                
#                 updates.index_add_(0, limited_indices, weighted_output)
        
#         routed_output = updates.view(batch_size, seq_len, -1)
        
#         # Process through shared expert
#         shared_output = self.shared_expert(x)
        
#         # Combine outputs with learnable weights
#         weights = F.softmax(torch.stack([self.routed_weight, self.shared_weight]), dim=0)
#         final_output = weights[0] * routed_output + weights[1] * shared_output
        
#         return final_output, load_balance_loss

# class SwitchTransformer(nn.Module):
#     def __init__(self, n_embed, num_experts):
#         super(SwitchTransformer, self).__init__()
#         self.router = nn.Linear(n_embed, num_experts)
#         self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
#         self.num_experts = num_experts
        
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
        
#         # Router logic - select single expert
#         router_logits = self.router(x)  # (B, T, num_experts)
#         expert_weights = F.softmax(router_logits, dim=-1)
#         expert_choice = torch.argmax(expert_weights, dim=-1)  # (B, T)
        
#         # Initialize output tensor
#         final_output = torch.zeros_like(x)
        
#         # Process through selected experts
#         for i in range(self.num_experts):
#             # Create mask for current expert
#             expert_mask = (expert_choice == i)  # (B, T)
            
#             if expert_mask.any():
#                 # Get inputs for current expert
#                 expert_input = x[expert_mask]  # (selected_tokens, C)
                
#                 # Process through expert
#                 expert_output = self.experts[i](expert_input)
                
#                 # Get weights for current expert
#                 expert_weights_i = expert_weights[expert_mask, i].unsqueeze(-1)
                
#                 # Add weighted output to final result
#                 final_output[expert_mask] = expert_output * expert_weights_i
        
#         return final_output