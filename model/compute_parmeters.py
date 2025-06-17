import torch.nn as nn 
from moe.MoE import Expert
from moe.DeepSeekMoE import FGExpert
class ModelParameterCount:
    def __init__(self, Module: nn.Module, config):
        super().__init__()
        self.model = Module(config)
        self.top_k = config.top_k_experts if config.top_k_experts is not None else 0 
        self.n_layers = config.n_layers
        total_params, expert_params = self.count_active_parameters()
        print(f"{total_params/1e6}M - A{expert_params/1e6}M")

    def count_parameters(self, module):
        # Chỉ gọi .named_parameters() nếu module là nn.Module
        if isinstance(module, nn.Module):
            return sum(param.numel() for _, param in module.named_parameters())
        else:
            # Nếu là Parameter, trả về số phần tử
            return module.numel()
    def count_active_parameters(self):
        shared_params = 0
        if self.top_k == 0: 
            print("Dense model")
        num_experts = 0
        expert_params = 0
        total_params = 0
        nonactive_experts = 0
        for name, param in self.model.named_parameters():
            if 'experts' not in name:  # hoặc tên module chứa expert
                shared_params += param.numel()
        for _, module in self.model.named_modules():
            if num_experts < self.top_k * self.n_layers and ( isinstance(module, Expert) or isinstance(module, FGExpert) ) : 
                #print(module)
                expert_params += self.count_parameters(module)
                #print(expert_params)
                num_experts += 1 
            elif isinstance(module, Expert) or isinstance(module, FGExpert):
                nonactive_experts +=1
        print(shared_params)
        active_params = shared_params + expert_params
        print(f"num experts: {(num_experts + nonactive_experts)/self.n_layers}")
        return self.count_parameters(self.model), active_params 