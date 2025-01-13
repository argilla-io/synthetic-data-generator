# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "synthetic-dataset-generator",
# ]
# ///
# vllm serve Qwen/Qwen2.5-1.5B-Instruct
import os

from synthetic_dataset_generator import launch

os.environ["HF_TOKEN"] = "hf_..."  # push the data to huggingface
os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:8000/"  # vllm base url
os.environ["MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"  # model id
os.environ["TOKENIZER_ID"] = "Qwen/Qwen2.5-1.5B-Instruct"  # tokenizer id
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "qwen2"
os.environ["MAX_NUM_ROWS"] = "10000"
os.environ["DEFAULT_BATCH_SIZE"] = "2"
os.environ["MAX_NUM_TOKENS"] = "1024"

launch()
