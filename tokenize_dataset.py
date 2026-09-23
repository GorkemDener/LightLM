# python tokenize_dataset.py
#
# Tokenizes Cosmopedia v2 (the "smollm-corpus" cosmopedia-v2 config, ~28B tokens,
# 104 parquet files) into the datatrove shard format (.ds files) that trainer.py's
# DataLoader expects. Reads from a LOCAL copy of the raw parquet files -- download
# them first with:
#
#   hf download HuggingFaceTB/smollm-corpus --repo-type dataset \
#       --include "cosmopedia-v2/*" --local-dir /fast/ndener/lightlm_workbench/raw_data/smollm-corpus
#
# Run this where you have real CPUs/disk (an interactive Condor job), not on a login
# node -- login nodes are capped at 50% of one CPU core.

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer
from transformers import AutoTokenizer

TOKENIZER_ID = "HuggingFaceTB/SmolLM-360M"   # same tokenizer used in train.py/ddp_train.py
RAW_DATA_DIR = "/fast/ndener/lightlm_workbench/raw_data/smollm-corpus/cosmopedia-v2"
OUTPUT_DIR = "/fast/ndener/lightlm_workbench/tokenized/cosmopedia"
LOGGING_DIR = "/fast/ndener/lightlm_workbench/tokenized/logs"
NUM_TASKS = 15   # match request_cpus in _condor_submit_debug.sub

eos_token = AutoTokenizer.from_pretrained(TOKENIZER_ID).eos_token

executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=RAW_DATA_DIR,
            glob_pattern="*.parquet",
        ),
        DocumentTokenizer(
            output_folder=OUTPUT_DIR,
            tokenizer_name_or_path=TOKENIZER_ID,
            eos_token=eos_token,
            shuffle_documents=True,
        ),
    ],
    tasks=NUM_TASKS,
    logging_dir=LOGGING_DIR,
)

if __name__ == "__main__":
    executor.run()
