# pip install synthetic-dataset-generator
# ollama serve
# ollama run llama3.1:8b-instruct-q8_0
import os

from synthetic_dataset_generator import launch

# os.environ["HF_TOKEN"] = "hf_..."  # push the data to huggingface
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434/"  # ollama base url
os.environ["MODEL"] = "qwen2.5:32b-instruct-q5_K_S"  # model id
os.environ["TOKENIZER_ID"] = "Qwen/Qwen2.5-32B-Instruct"  # tokenizer id
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "qwen2"
os.environ["MAX_NUM_ROWS"] = "10000"
os.environ["DEFAULT_BATCH_SIZE"] = "2"
os.environ["MAX_NUM_TOKENS"] = "1024"

launch()
