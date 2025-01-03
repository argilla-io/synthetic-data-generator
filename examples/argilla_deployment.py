# pip install synthetic-dataset-generator
import os

from synthetic_dataset_generator import launch

# Follow https://docs.argilla.io/latest/getting_started/quickstart/ to get your Argilla API key and URL
os.environ["HF_TOKEN"] = "hf_..."
os.environ["ARGILLA_API_URL"] = (
    "https://[your-owner-name]-[your_space_name].hf.space"  # argilla base url
)
os.environ["ARGILLA_API_KEY"] = "my_api_key"  # argilla api key

launch()
