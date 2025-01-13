# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "synthetic-dataset-generator",
# ]
# ///

import os

from synthetic_dataset_generator import launch

os.environ["HF_TOKEN"] = "hf_..."  # push the data to huggingface
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1/"  # openai base url
os.environ["API_KEY"] = os.getenv("OPENAI_API_KEY")  # openai api key
os.environ["MODEL"] = "gpt-4o"  # model id
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = None  # chat data not supported with OpenAI

launch()
