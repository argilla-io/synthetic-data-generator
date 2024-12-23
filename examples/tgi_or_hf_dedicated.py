# pip install synthetic-dataset-generator
import os

from synthetic_dataset_generator import launch

assert os.getenv("HF_TOKEN")  # push the data to huggingface
os.environ["HUGGINGFACE_BASE_URL"] = "http://127.0.0.1:3000/"
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "llama3"
os.environ["TOKENIZER_ID"] = (
    "meta-llama/Llama-3.1-8B-Instruct"  # tokenizer for model hosted on endpoint
)
os.environ["MODEL"] = None  # model is linked to endpoint

launch()
