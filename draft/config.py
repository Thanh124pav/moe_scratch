# Configuration for MoE Transformer

# Model Architecture
VOCAB_SIZE = 50000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DFF = 2048
DROPOUT_RATE = 0.1

# MoE Parameters
NUM_EXPERTS = 8
EXPERTS_PER_TOKEN = 2
EXPERT_CAPACITY_FACTOR = 1.0
# Expert Capacity Adaptation
EXPERT_CAPACITY_ADAPTATION = True
MIN_EXPERT_CAPACITY = 0.5
MAX_EXPERT_CAPACITY = 2.0
ADAPTATION_INTERVAL = 1000  # Steps between capacity adjustments

# Context-Aware Routing
CONTEXT_AWARE_ROUTING = True
CONTEXT_WINDOW_SIZE = 3  # Number of neighboring tokens to consider

# Hierarchical MoE
HIERARCHICAL_MOE = True
NUM_LEVELS = 2  # Number of hierarchy levels
EXPERTS_PER_LEVEL = [4, 2]  # Number of experts at each level (length should match NUM_LEVELS)

# Expert Specialization Regularization
SPECIALIZATION_REGULARIZATION = True
SPECIALIZATION_LOSS_WEIGHT = 0.005  # Weight for specialization loss

# Training
BATCH_SIZE = 64
MAX_SEQ_LEN = 128
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
WARMUP_STEPS = 4000

# Router
ROUTER_TYPE = 'gumbel'  # Options: 'gumbel', 'softmax'
LOAD_BALANCE_LOSS_WEIGHT = 0.01

# Checkpointing
CHECKPOINT_PATH = './checkpoints/moe_transformer' 