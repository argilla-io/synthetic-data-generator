import math
import random

from distilabel.models import ClientvLLM, InferenceEndpointsLLM, OllamaLLM, OpenAILLM
from distilabel.steps.tasks import TextGeneration

from synthetic_dataset_generator.constants import (
    API_KEYS,
    DEFAULT_BATCH_SIZE,
    HUGGINGFACE_BASE_URL,
    MODEL,
    OLLAMA_BASE_URL,
    OPENAI_BASE_URL,
    TOKENIZER_ID,
    VLLM_BASE_URL,
)

TOKEN_INDEX = 0


def _get_next_api_key():
    global TOKEN_INDEX
    api_key = API_KEYS[TOKEN_INDEX % len(API_KEYS)]
    TOKEN_INDEX += 1
    return api_key


def _get_prompt_rewriter():
    generation_kwargs = {
        "temperature": 1,
    }
    system_prompt = "You are a prompt rewriter. You are given a prompt and you need to rewrite it keeping the same structure but highlighting different aspects of the original without adding anything new."
    prompt_rewriter = TextGeneration(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        system_prompt=system_prompt,
        use_system_prompt=True,
    )
    prompt_rewriter.load()
    return prompt_rewriter


def get_rewritten_prompts(prompt: str, num_rows: int):
    prompt_rewriter = _get_prompt_rewriter()
    # create prompt rewrites
    inputs = [
        {"instruction": f"Original prompt: {prompt} \nRewritten prompt: "}
        for i in range(math.floor(num_rows / 100))
    ]
    n_processed = 0
    prompt_rewrites = [prompt]
    while n_processed < num_rows:
        batch = list(
            prompt_rewriter.process(
                inputs=inputs[n_processed : n_processed + DEFAULT_BATCH_SIZE]
            )
        )
        prompt_rewrites += [entry["generation"] for entry in batch[0]]
        n_processed += DEFAULT_BATCH_SIZE
        random.seed(a=random.randint(0, 2**32 - 1))
    return prompt_rewrites


def _get_llm_class() -> str:
    if OPENAI_BASE_URL:
        return "OpenAILLM"
    elif OLLAMA_BASE_URL:
        return "OllamaLLM"
    elif HUGGINGFACE_BASE_URL:
        return "InferenceEndpointsLLM"
    elif VLLM_BASE_URL:
        return "ClientvLLM"
    else:
        return "InferenceEndpointsLLM"


def _get_llm(use_magpie_template=False, **kwargs):
    if OPENAI_BASE_URL:
        llm = OpenAILLM(
            model=MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=_get_next_api_key(),
            **kwargs,
        )
        if "generation_kwargs" in kwargs:
            if "stop_sequences" in kwargs["generation_kwargs"]:
                kwargs["generation_kwargs"]["stop"] = kwargs["generation_kwargs"][
                    "stop_sequences"
                ]
                del kwargs["generation_kwargs"]["stop_sequences"]
            if "do_sample" in kwargs["generation_kwargs"]:
                del kwargs["generation_kwargs"]["do_sample"]
    elif OLLAMA_BASE_URL:
        if "generation_kwargs" in kwargs:
            if "max_new_tokens" in kwargs["generation_kwargs"]:
                kwargs["generation_kwargs"]["num_predict"] = kwargs[
                    "generation_kwargs"
                ]["max_new_tokens"]
                del kwargs["generation_kwargs"]["max_new_tokens"]
            if "stop_sequences" in kwargs["generation_kwargs"]:
                kwargs["generation_kwargs"]["stop"] = kwargs["generation_kwargs"][
                    "stop_sequences"
                ]
                del kwargs["generation_kwargs"]["stop_sequences"]
            if "do_sample" in kwargs["generation_kwargs"]:
                del kwargs["generation_kwargs"]["do_sample"]
            options = kwargs["generation_kwargs"]
            del kwargs["generation_kwargs"]
            kwargs["generation_kwargs"] = {}
            kwargs["generation_kwargs"]["options"] = options
        llm = OllamaLLM(
            model=MODEL,
            host=OLLAMA_BASE_URL,
            tokenizer_id=TOKENIZER_ID or MODEL,
            use_magpie_template=use_magpie_template,
            **kwargs,
        )
    elif HUGGINGFACE_BASE_URL:
        kwargs["generation_kwargs"]["do_sample"] = True
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            base_url=HUGGINGFACE_BASE_URL,
            tokenizer_id=TOKENIZER_ID or MODEL,
            use_magpie_template=use_magpie_template,
            **kwargs,
        )
    elif VLLM_BASE_URL:
        if "generation_kwargs" in kwargs:
            if "do_sample" in kwargs["generation_kwargs"]:
                del kwargs["generation_kwargs"]["do_sample"]
        llm = ClientvLLM(
            base_url=VLLM_BASE_URL,
            model=MODEL,
            tokenizer=TOKENIZER_ID or MODEL,
            api_key=_get_next_api_key(),
            use_magpie_template=use_magpie_template,
            **kwargs,
        )
    else:
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            tokenizer_id=TOKENIZER_ID or MODEL,
            model_id=MODEL,
            use_magpie_template=use_magpie_template,
            **kwargs,
        )

    return llm


try:
    llm = _get_llm()
    llm.load()
    llm.generate([[{"content": "Hello, world!", "role": "user"}]])
except Exception as e:
    raise Exception(f"Error loading {llm.__class__.__name__}: {e}")
