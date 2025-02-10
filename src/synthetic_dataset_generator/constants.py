import os
import warnings

import argilla as rg

# Inference
MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", 2048))
MAX_NUM_ROWS = int(os.getenv("MAX_NUM_ROWS", 1000))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", 5))

# Directory for outputs
SAVE_LOCAL_DIR = os.getenv(key="SAVE_LOCAL_DIR", default=None)

# Models
MODEL = os.getenv("MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TOKENIZER_ID = os.getenv(key="TOKENIZER_ID", default=None)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
HUGGINGFACE_BASE_URL = os.getenv("HUGGINGFACE_BASE_URL")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")

# Just used in case of selecting a different model for completions
MODEL_COMPLETION = os.getenv("MODEL_COMPLETION", MODEL)
TOKENIZER_ID_COMPLETION = os.getenv("TOKENIZER_ID_COMPLETION", TOKENIZER_ID)
OPENAI_BASE_URL_COMPLETION = os.getenv("OPENAI_BASE_URL_COMPLETION", OPENAI_BASE_URL)
OLLAMA_BASE_URL_COMPLETION = os.getenv("OLLAMA_BASE_URL_COMPLETION", OLLAMA_BASE_URL)
HUGGINGFACE_BASE_URL_COMPLETION = os.getenv(
    "HUGGINGFACE_BASE_URL_COMPLETION", HUGGINGFACE_BASE_URL
)
VLLM_BASE_URL_COMPLETION = os.getenv("VLLM_BASE_URL_COMPLETION", VLLM_BASE_URL)

base_urls = [OPENAI_BASE_URL, OLLAMA_BASE_URL, HUGGINGFACE_BASE_URL, VLLM_BASE_URL]
base_urls_completion = [
    OPENAI_BASE_URL_COMPLETION,
    OLLAMA_BASE_URL_COMPLETION,
    HUGGINGFACE_BASE_URL_COMPLETION,
    VLLM_BASE_URL_COMPLETION,
]


# Validate the configuration of the model and base URLs.
def validate_configuration(base_urls, model, env_context=""):
    huggingface_url = base_urls[2]
    if huggingface_url and model:
        raise ValueError(
            f"`HUGGINGFACE_BASE_URL{env_context}` and `MODEL{env_context}` cannot be set at the same time. "
            "Use a model id for serverless inference and a base URL dedicated to Hugging Face Inference Endpoints."
        )

    if not model and any(base_urls):
        raise ValueError(
            f"`MODEL{env_context}` is not set. Please provide a model id for inference."
        )

    active_urls = [url for url in base_urls if url]
    if len(active_urls) > 1:
        raise ValueError(
            f"Multiple base URLs are provided: {', '.join(active_urls)}. "
            "Only one base URL can be set at a time."
        )
validate_configuration(base_urls, MODEL)
validate_configuration(base_urls_completion, MODEL_COMPLETION, "_COMPLETION")

BASE_URL = OPENAI_BASE_URL or OLLAMA_BASE_URL or HUGGINGFACE_BASE_URL or VLLM_BASE_URL
BASE_URL_COMPLETION = (
    OPENAI_BASE_URL_COMPLETION
    or OLLAMA_BASE_URL_COMPLETION
    or HUGGINGFACE_BASE_URL_COMPLETION
    or VLLM_BASE_URL_COMPLETION
)

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set. Ensure you have set the HF_TOKEN environment variable that has access to the Hugging Face Hub repositories and Inference Endpoints."
    )

_API_KEY = os.getenv("API_KEY")
API_KEYS = (
    [_API_KEY]
    if _API_KEY
    else [HF_TOKEN] + [os.getenv(f"HF_TOKEN_{i}") for i in range(1, 10)]
)
API_KEYS = [token for token in API_KEYS if token]

# Determine if SFT is available
SFT_AVAILABLE = False
llama_options = ["llama3", "llama-3", "llama 3"]
qwen_options = ["qwen2", "qwen-2", "qwen 2"]

if passed_pre_query_template := os.getenv("MAGPIE_PRE_QUERY_TEMPLATE", "").lower():
    SFT_AVAILABLE = True
    if passed_pre_query_template in llama_options:
        MAGPIE_PRE_QUERY_TEMPLATE = "llama3"
    elif passed_pre_query_template in qwen_options:
        MAGPIE_PRE_QUERY_TEMPLATE = "qwen2"
    else:
        MAGPIE_PRE_QUERY_TEMPLATE = passed_pre_query_template
elif MODEL.lower() in llama_options or any(
    option in MODEL.lower() for option in llama_options
):
    SFT_AVAILABLE = True
    MAGPIE_PRE_QUERY_TEMPLATE = "llama3"
elif MODEL.lower() in qwen_options or any(
    option in MODEL.lower() for option in qwen_options
):
    SFT_AVAILABLE = True
    MAGPIE_PRE_QUERY_TEMPLATE = "qwen2"

if OPENAI_BASE_URL:
    SFT_AVAILABLE = False

if not SFT_AVAILABLE:
    warnings.warn(
        "`SFT_AVAILABLE` is set to `False`. Use Hugging Face Inference Endpoints or Ollama to generate chat data, provide a `TOKENIZER_ID` and `MAGPIE_PRE_QUERY_TEMPLATE`. You can also use `HUGGINGFACE_BASE_URL` to with vllm."
    )
    MAGPIE_PRE_QUERY_TEMPLATE = None

# Embeddings
STATIC_EMBEDDING_MODEL = "minishlab/potion-base-8M"

# Argilla
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL") or os.getenv(
    "ARGILLA_API_URL_SDG_REVIEWER"
)
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY") or os.getenv(
    "ARGILLA_API_KEY_SDG_REVIEWER"
)

if not ARGILLA_API_URL or not ARGILLA_API_KEY:
    warnings.warn("ARGILLA_API_URL or ARGILLA_API_KEY is not set or is empty")
    argilla_client = None
else:
    argilla_client = rg.Argilla(
        api_url=ARGILLA_API_URL,
        api_key=ARGILLA_API_KEY,
    )
