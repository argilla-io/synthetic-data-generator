---
title: Synthetic Data Generator
short_description: Build datasets using natural language
emoji: 🧬
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.8.0
app_file: app.py
pinned: true
license: apache-2.0
hf_oauth: true
#header: mini
hf_oauth_scopes:
- read-repos
- write-repos
- manage-repos
- inference-api
---
<br>

<h2 align="center">
  <a href=""><img src="https://raw.githubusercontent.com/argilla-io/synthetic-data-generator/main/assets/logo.svg" alt="Synthetic Data Generator Logo" width="80%"></a>
</h2>
<h3 align="center">Build datasets using natural language</h3>

![Synthetic Data Generator](https://huggingface.co/spaces/argilla/synthetic-data-generator/resolve/main/assets/ui-full.png)

## Introduction

Synthetic Data Generator is a tool that allows you to create high-quality datasets for training and fine-tuning language models. It leverages the power of distilabel and LLMs to generate synthetic data tailored to your specific needs. [The announcement blog](https://huggingface.co/blog/synthetic-data-generator) goes over a practical example of how to use it but you can also watch the [video](https://www.youtube.com/watch?v=nXjVtnGeEss) to see it in action.

Supported Tasks:

- Text Classification
- Chat Data for Supervised Fine-Tuning
- Retrieval Augmented Generation

This tool simplifies the process of creating custom datasets, enabling you to:

- Describe the characteristics of your desired application
- Iterate on sample datasets
- Produce full-scale datasets
- Push your datasets to the [Hugging Face Hub](https://huggingface.co/datasets?other=datacraft) and/or [Argilla](https://docs.argilla.io/)

By using the Synthetic Data Generator, you can rapidly prototype and create datasets for, accelerating your AI development process.

<p align="center">
<a href="https://twitter.com/argilla_io">
<img src="https://img.shields.io/badge/twitter-black?logo=x"/>
</a>
<a href="https://www.linkedin.com/company/argilla-io">
<img src="https://img.shields.io/badge/linkedin-blue?logo=linkedin"/>
</a>
<a href="http://hf.co/join/discord">
<img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
</a>
</p>

## Installation

You can simply install the package with:

```bash
pip install synthetic-dataset-generator
```

### Quickstart

```python
from synthetic_dataset_generator import launch

launch()
```

### Environment Variables

- `HF_TOKEN`: Your [Hugging Face token](https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&tokenType=fineGrained) to push your datasets to the Hugging Face Hub and generate free completions from Hugging Face Inference Endpoints. You can find some configuration examples in the [examples](examples/) folder.

You can set the following environment variables to customize the generation process.

- `MAX_NUM_TOKENS`: The maximum number of tokens to generate, defaults to `2048`.
- `MAX_NUM_ROWS`: The maximum number of rows to generate, defaults to `1000`.
- `DEFAULT_BATCH_SIZE`: The default batch size to use for generating the dataset, defaults to `5`.

Optionally, you can use different API providers and models.

- `MODEL`: The model to use for generating the dataset, e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`, `gpt-4o`, `llama3.1`.
- `API_KEY`: The API key to use for the generation API, e.g. `hf_...`, `sk-...`. If not provided, it will default to the provided `HF_TOKEN` environment variable.
- `OPENAI_BASE_URL`: The base URL for any OpenAI compatible API, e.g. `https://api.openai.com/v1/`.
- `OLLAMA_BASE_URL`: The base URL for any Ollama compatible API, e.g. `http://127.0.0.1:11434/`.
- `HUGGINGFACE_BASE_URL`: The base URL for any Hugging Face compatible API, e.g. TGI server or Dedicated Inference Endpoints. If you want to use serverless inference, only set the `MODEL`.
- `VLLM_BASE_URL`: The base URL for any VLLM compatible API, e.g. `http://localhost:8000/`.

SFT and Chat Data generation is not supported with OpenAI Endpoints. Additionally, you need to configure it per model family based on their prompt templates using the right `TOKENIZER_ID` and `MAGPIE_PRE_QUERY_TEMPLATE` environment variables.

- `TOKENIZER_ID`: The tokenizer ID to use for the magpie pipeline, e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- `MAGPIE_PRE_QUERY_TEMPLATE`: Enforce setting the pre-query template for Magpie, which is only supported with Hugging Face Inference Endpoints. `llama3` and `qwen2` are supported out of the box and will use `"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"` and `"<|im_start|>user\n"`, respectively. For other models, you can pass a custom pre-query template string.

Optionally, you can also push your datasets to Argilla for further curation by setting the following environment variables:

- `ARGILLA_API_KEY`: Your Argilla API key to push your datasets to Argilla.
- `ARGILLA_API_URL`: Your Argilla API URL to push your datasets to Argilla.

### Argilla integration

Argilla is an open source tool for data curation. It allows you to annotate and review datasets, and push curated datasets to the Hugging Face Hub. You can easily get started with Argilla by following the [quickstart guide](https://docs.argilla.io/latest/getting_started/quickstart/).

![Argilla integration](https://huggingface.co/spaces/argilla/synthetic-data-generator/resolve/main/assets/argilla.png)

## Custom synthetic data generation?

Each pipeline is based on distilabel, so you can easily change the LLM or the pipeline steps.

Check out the [distilabel library](https://github.com/argilla-io/distilabel) for more information.

## Development

Install the dependencies:

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the dependencies
pip install -e . # pdm install
```

Run the app:

```bash
python app.py
```
