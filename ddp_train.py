# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
# torchrun --standalone --nproc_per_node=4 ddp_train.py

from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader

from transformers import AutoTokenizer
import torch
from torch.distributed import init_process_group, destroy_process_group

import os

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

# Same "MoE 2+1" recipe as train.py, just spread across world_size GPUs:
# batch_size stays at the per-GPU memory ceiling (32) and accumulation_steps
# is divided by world_size so batch_size * max_seq_len * accumulation_steps *
# world_size still equals 262,144 tokens/step -- identical training math to
# the single-GPU run, just parallelized.
train_config = TrainerConfig(
    vocab_size = tokenizer.vocab_size,
    num_epochs = 1,

    use_ddp = True,
    use_moe = True,
    use_lossfreebalance = True,
    clean_cuda_cache = True,
    use_compile = True,
    use_dtype = "bfloat16",

    seed = 1338,
    max_seq_len = 1024, # matches the published "MoE 2+1" context_len
    batch_size = 32,
    accumulation_steps = 2, # 8 (single-GPU) / 4 GPUs

    weight_decay = 0.1,
    warmup_ratio = 0.1,
    learning_rate = 4e-4,
    betas = (0.90, 0.97),
    update_rate = 5e-6,

    val_ratio = 0.005,
    steps_for_eval = 20,
    eval_interval = 5000,

    checkpoints_frequency = 5000,
    path_to_checkpoints = "./model_testing",

    tokenized_dataset_path = "/fast/ndener/lightlm_workbench/tokenized/cosmopedia",
    eval_log_file = "log/eval_cosmopedia.txt",
)

config = ModelConfig(
        vocab_size = tokenizer.vocab_size,

        num_dims = 384,
        num_heads = 16,
        num_kv_heads = 4,
        num_layers = 32,
        ffn_hidden_dims = 1024,

        rmsnorm_eps = 1e-6,
        rope_theta = 1e5,

        context_len = 1024,

        use_cache = False,
        use_flash = True,
        use_moe = True,

        moe_num_experts = 2,
        moe_active_experts = 2,
        moe_eps = 1e-6,
        moe_aux_loss_coef = 0.01,
        moe_shared_experts = 1,
        use_lossfreebalance = True,
    )

init_process_group("nccl")

model = Transformer(config)

data_loader = DataLoader(train_config, int(os.environ['RANK']), 
                         int(os.environ['WORLD_SIZE']))


trainer = Trainer(train_config, model, tokenizer)
trainer.train(data_loader)

destroy_process_group()