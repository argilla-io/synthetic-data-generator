# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "synthetic-dataset-generator",
# ]
# ///
# ollama serve
# ollama run llama3.2
# ollama run llama3.2:1b
import os

from synthetic_dataset_generator import launch

os.environ["OLLAMA_BASE_URL"] = (
    "http://127.0.0.1:11434/"  # in this case, the same base url for both models
)

os.environ["MODEL"] = "llama3.2" # model for instruction generation
os.environ["MODEL_COMPLETION"] = "llama3.2:1b" # model for completion generation

os.environ["TOKENIZER_ID"] = "meta-llama/Llama-3.2-3B-Instruct" # tokenizer for instruction generation
os.environ["TOKENIZER_ID_COMPLETION"] = "meta-llama/Llama-3.2-1B-Instruct" # tokenizer for completion generation

os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "llama3" # magpie template required for instruction generation

launch()
