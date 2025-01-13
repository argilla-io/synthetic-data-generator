from datasets import get_dataset_config_names, get_dataset_split_names
from distilabel.models import InferenceEndpointsLLM
from distilabel.steps.tasks import (
    TextGeneration,
    UltraFeedback,
)

from synthetic_dataset_generator.constants import BASE_URL, MAX_NUM_TOKENS, MODEL
from synthetic_dataset_generator.pipelines.base import _get_next_api_key
from synthetic_dataset_generator.utils import extract_column_names


def get_ultrafeedback_evaluator(aspect, is_sample):
    ultrafeedback_evaluator = UltraFeedback(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            base_url=BASE_URL,
            api_key=_get_next_api_key(),
            generation_kwargs={
                "temperature": 0.01,
                "max_new_tokens": MAX_NUM_TOKENS if not is_sample else 512,
            },
        ),
        aspect=aspect,
    )
    ultrafeedback_evaluator.load()
    return ultrafeedback_evaluator


def get_custom_evaluator(prompt_template, structured_output, columns, is_sample):
    custom_evaluator = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            base_url=BASE_URL,
            api_key=_get_next_api_key(),
            structured_output={"format": "json", "schema": structured_output},
            generation_kwargs={
                "temperature": 0.01,
                "max_new_tokens": MAX_NUM_TOKENS if not is_sample else 512,
            },
        ),
        template=prompt_template,
        columns=columns,
    )
    custom_evaluator.load()
    return custom_evaluator


def generate_ultrafeedback_pipeline_code(
    repo_id, subset, split, aspects, instruction_column, response_columns, num_rows
):
    if len(aspects) == 1:
        code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
import os
from datasets import load_dataset
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import UltraFeedback
from distilabel.models import InferenceEndpointsLLM

MODEL = "{MODEL}"
BASE_URL = "{BASE_URL}"
os.environ["API_KEY"] = "hf_xxx" # https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&canReadGatedRepos=true&tokenType=fineGrained

hf_ds = load_dataset("{repo_id}", "{subset}", split="{split}[:{num_rows}]")
data = preprocess_data(hf_ds, "{instruction_column}", "{response_columns}") # to get a list of dictionaries

with Pipeline(name="ultrafeedback") as pipeline:

    load_the_dataset = LoadDataFromDicts(
        data = data,
    )

    ultrafeedback_evaluator = UltraFeedback(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            base_url=BASE_URL,
            api_key=os.environ["API_KEY"],
            generation_kwargs={{
                "temperature": 0.01,
                "max_new_tokens": {MAX_NUM_TOKENS},
            }},
        ),
        aspect=aspect,
    )

    load_the_dataset >> ultrafeedback_evaluator

if __name__ == "__main__":
    distiset = pipeline.run()
"""
    else:
        code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
import os
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, CombineOutputs
from distilabel.steps.tasks import UltraFeedback
from distilabel.models import InferenceEndpointsLLM

MODEL = "{MODEL}"
BASE_URL = "{BASE_URL}"
os.environ["BASE_URL"] = "hf_xxx" # https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&canReadGatedRepos=true&tokenType=fineGrained

hf_ds = load_dataset("{repo_id}", "{subset}", split="{split}")
data = preprocess_data(hf_ds, "{instruction_column}", "{response_columns}") # to get a list of dictionaries

with Pipeline(name="ultrafeedback") as pipeline:

    load_the_dataset = LoadDataFromDicts(
        data = data,
    )

    tasks = []
    for aspect in aspects:
        evaluate_responses = UltraFeedback(
            name=f"evaluate-responses-{{aspect}}",
            aspect=aspect,
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                base_url=BASE_URL,
                api_key=os.environ["BASE_URL"],
                generation_kwargs={{
                    "temperature": 0.01,
                    "max_new_tokens": {MAX_NUM_TOKENS},
                }},
            output_mappings={{
                "ratings": f"ratings_{{aspect}}",
                "types": f"type_{{aspect}}",
                "rationales": f"rationales_for_types_{{aspect}}",
                "rationales-for-ratings": f"rationales_for_ratings_{{aspect}}",
            }} if aspect in ["truthfulness", "helpfulness"] else {{"rationales": f"rationales_{{aspect}}", "ratings": f"ratings_{{aspect}}"}},
        )
        tasks.append(evaluate_responses)

    combine_outputs = CombineOutputs()

    load_the_dataset >> tasks >> combine_outputs

if __name__ == "__main__":
    distiset = pipeline.run()
"""
    return code


def generate_custom_pipeline_code(
    repo_id, subset, split, prompt_template, structured_output, num_rows
):
    columns = extract_column_names(structured_output)
    code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints, instructor]`
import os
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration
from distilabel.models import InferenceEndpointsLLM

MODEL = "{MODEL}"
BASE_URL = "{BASE_URL}"
CUSTOM_TEMPLATE = "{prompt_template}"
os.environ["HF_TOKEN"] = "hf_xxx" # https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&canReadGatedRepos=true&tokenType=fineGrained

with Pipeline(name="custom-evaluation") as pipeline:
    load_the_dataset = LoadDataFromHub(
        repo_id="{repo_id}",
        config="{subset}",
        split="{split}",
        num_examples={num_rows},
        batch_size=2
    )
    custom_evaluator = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            base_url=BASE_URL,
            api_key=os.environ["HF_TOKEN"],
            structured_output={{"format": "json", "schema": {structured_output}}},
            generation_kwargs={{
                "temperature": 0.01,
                "max_new_tokens": {MAX_NUM_TOKENS},
            }},
        ),
        template=CUSTOM_TEMPLATE,
        columns={columns}
    )

    load_the_dataset >> custom_evaluator

if __name__ == "__main__":
    distiset = pipeline.run()
"""
    return code


def generate_pipeline_code(
    repo_id,
    aspects,
    instruction_column,
    response_columns,
    prompt_template,
    structured_output,
    num_rows,
    eval_type,
):
    if repo_id is None:
        subset = "default"
        split = "train"
    else:
        subset = get_dataset_config_names(repo_id)[0]
        split = get_dataset_split_names(repo_id, subset)[0]
    if eval_type == "ultrafeedback":
        return generate_ultrafeedback_pipeline_code(
            repo_id,
            subset,
            split,
            aspects,
            instruction_column,
            response_columns,
            num_rows,
        )
    return generate_custom_pipeline_code(
        repo_id, subset, split, prompt_template, structured_output, num_rows
    )
