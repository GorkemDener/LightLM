# python convert_checkpoint.py <checkpoint.pt> <output_dir>
#
# Trainer.save_checkpoints() writes a raw {'model': state_dict, 'optimizer': ...} .pt
# file. Transformer.from_pretrained() (used by eval_model.py / test_outputs.py) expects
# a save_pretrained()-style directory (config.json + model.safetensors) instead. This
# bridges the two: loads a raw training checkpoint and re-saves it in that format.
#
# Must edit CONFIG below to match whatever ModelConfig train.py actually used to
# produce the checkpoint (e.g. the published "MoE 2+1" settings).

import sys
import torch

from model import Transformer, ModelConfig

CONFIG = ModelConfig(
    vocab_size=49152,

    num_dims=384,
    num_heads=16,
    num_kv_heads=4,
    num_layers=32,
    ffn_hidden_dims=1024,

    rmsnorm_eps=1e-6,
    rope_theta=1e5,

    context_len=1024,

    use_cache=False,
    use_flash=True,
    use_moe=True,

    moe_num_experts=2,
    moe_active_experts=2,
    moe_eps=1e-6,
    moe_aux_loss_coef=0.01,
    moe_shared_experts=1,
    use_lossfreebalance=True,
)


def convert(checkpoint_path: str, output_dir: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]

    # strip torch.compile()'s "_orig_mod." prefix and DDP's "module." prefix, if present.
    # A DDP + compiled checkpoint stacks both ("module._orig_mod.<name>"), so keep
    # stripping until neither prefix matches instead of checking each only once.
    new_state_dict = {}
    for k, v in state_dict.items():
        while True:
            if k.startswith("module."):
                k = k[len("module."):]
            elif k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            else:
                break
        new_state_dict[k] = v

    model = Transformer(CONFIG)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys not found in checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint, ignored: {unexpected}")

    model.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <checkpoint.pt> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
