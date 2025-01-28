# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "synthetic-dataset-generator",
# ]
# ///
import os

from synthetic_dataset_generator import launch

os.environ["HF_TOKEN"] = "hf_..."  # push the data to huggingface
os.environ["MODEL"] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # use model for instructions
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "<｜begin▁of▁sentence｜>User: "  # use the custom template for the model


launch()
