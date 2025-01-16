import ast
import random
import uuid
from typing import Dict, List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import Dataset
from distilabel.distiset import Distiset
from huggingface_hub import HfApi

from synthetic_dataset_generator.apps.base import (
    combine_datasets,
    hide_success_message,
    push_pipeline_code_to_hub,
    show_success_message,
    test_max_num_rows,
    validate_argilla_user_workspace_dataset,
    validate_push_to_hub,
)
from synthetic_dataset_generator.constants import (
    BASE_URL,
    DEFAULT_BATCH_SIZE,
    MODEL,
    SFT_AVAILABLE,
)
from synthetic_dataset_generator.pipelines.base import get_rewritten_prompts
from synthetic_dataset_generator.pipelines.chat import (
    DEFAULT_DATASET_DESCRIPTIONS,
    generate_pipeline_code,
    get_magpie_generator,
    get_prompt_generator,
    get_response_generator,
)
from synthetic_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from synthetic_dataset_generator.utils import (
    get_argilla_client,
    get_org_dropdown,
    get_random_repo_name,
    swap_visibility,
)


def convert_dataframe_messages(dataframe: pd.DataFrame) -> pd.DataFrame:
    def convert_to_list_of_dicts(messages: str) -> List[Dict[str, str]]:
        return ast.literal_eval(
            messages.replace("'user'}", "'user'},")
            .replace("'system'}", "'system'},")
            .replace("'assistant'}", "'assistant'},")
        )

    if "messages" in dataframe.columns:
        dataframe["messages"] = dataframe["messages"].apply(
            lambda x: convert_to_list_of_dicts(x) if isinstance(x, str) else x
        )
    return dataframe


def generate_system_prompt(dataset_description, progress=gr.Progress()):
    progress(0.1, desc="Initializing")
    generate_description = get_prompt_generator()
    progress(0.5, desc="Generating")
    result = next(
        generate_description.process(
            [
                {
                    "instruction": dataset_description,
                }
            ]
        )
    )[0]["generation"]
    progress(1.0, desc="Prompt generated")
    return result


def generate_sample_dataset(system_prompt, num_turns, progress=gr.Progress()):
    progress(0.1, desc="Generating sample dataset")
    dataframe = generate_dataset(
        system_prompt=system_prompt,
        num_turns=num_turns,
        num_rows=10,
        progress=progress,
        is_sample=True,
    )
    progress(1.0, desc="Sample dataset generated")
    return dataframe


def _get_dataframe():
    return gr.Dataframe(
        headers=["prompt", "completion"],
        wrap=True,
        interactive=False,
    )


def generate_dataset(
    system_prompt: str,
    num_turns: int = 1,
    num_rows: int = 10,
    temperature: float = 0.9,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pd.DataFrame:
    num_rows = test_max_num_rows(num_rows)
    progress(0.0, desc="(1/2) Generating instructions")
    magpie_generator = get_magpie_generator(
        system_prompt, num_turns, temperature, is_sample
    )
    response_generator = get_response_generator(
        system_prompt, num_turns, temperature, is_sample
    )
    total_steps: int = num_rows * 2
    batch_size = DEFAULT_BATCH_SIZE

    # create prompt rewrites
    prompt_rewrites = get_rewritten_prompts(system_prompt, num_rows)

    # create instructions
    n_processed = 0
    magpie_results = []
    while n_processed < num_rows:
        progress(
            0.5 * n_processed / num_rows,
            total=total_steps,
            desc="(1/2) Generating instructions",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        rewritten_system_prompt = random.choice(prompt_rewrites)
        inputs = [{"system_prompt": rewritten_system_prompt} for _ in range(batch_size)]
        batch = list(magpie_generator.process(inputs=inputs))
        magpie_results.extend(batch[0])
        n_processed += batch_size
        random.seed(a=random.randint(0, 2**32 - 1))
    progress(0.5, desc="(1/2) Generating instructions")

    # generate responses
    n_processed = 0
    response_results = []
    if num_turns == 1:
        while n_processed < num_rows:
            progress(
                0.5 + 0.5 * n_processed / num_rows,
                total=total_steps,
                desc="(2/2) Generating responses",
            )
            batch = magpie_results[n_processed : n_processed + batch_size]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses[0])
            n_processed += batch_size
            random.seed(a=random.randint(0, 2**32 - 1))
        for result in response_results:
            result["prompt"] = result["instruction"]
            result["completion"] = result["generation"]
            result["system_prompt"] = system_prompt
    else:
        for result in magpie_results:
            result["conversation"].insert(
                0, {"role": "system", "content": system_prompt}
            )
            result["messages"] = result["conversation"]
        while n_processed < num_rows:
            progress(
                0.5 + 0.5 * n_processed / num_rows,
                total=total_steps,
                desc="(2/2) Generating responses",
            )
            batch = magpie_results[n_processed : n_processed + batch_size]
            responses = list(response_generator.process(inputs=batch))
            response_results.extend(responses[0])
            n_processed += batch_size
            random.seed(a=random.randint(0, 2**32 - 1))
        for result in response_results:
            result["messages"].append(
                {"role": "assistant", "content": result["generation"]}
            )
    progress(
        1,
        total=total_steps,
        desc="(2/2) Creating dataset",
    )

    # create distiset
    distiset_results = []
    for result in response_results:
        record = {}
        for relevant_keys in [
            "messages",
            "prompt",
            "completion",
            "model_name",
            "system_prompt",
        ]:
            if relevant_keys in result:
                record[relevant_keys] = result[relevant_keys]
        distiset_results.append(record)

    distiset = Distiset(
        {
            "default": Dataset.from_list(distiset_results),
        }
    )

    # If not pushing to hub generate the dataset directly
    distiset = distiset["default"]
    if num_turns == 1:
        outputs = distiset.to_pandas()[["prompt", "completion", "system_prompt"]]
    else:
        outputs = distiset.to_pandas()[["messages"]]
    dataframe = pd.DataFrame(outputs)
    progress(1.0, desc="Dataset generation completed")
    return dataframe


def push_dataset_to_hub(
    dataframe: pd.DataFrame,
    org_name: str,
    repo_name: str,
    oauth_token: Union[gr.OAuthToken, None],
    private: bool,
    pipeline_code: str,
    progress=gr.Progress(),
):
    progress(0.0, desc="Validating")
    repo_id = validate_push_to_hub(org_name, repo_name)
    progress(0.3, desc="Converting")
    original_dataframe = dataframe.copy(deep=True)
    dataframe = convert_dataframe_messages(dataframe)
    progress(0.7, desc="Creating dataset")
    dataset = Dataset.from_pandas(dataframe)
    dataset = combine_datasets(repo_id, dataset, oauth_token)
    progress(0.9, desc="Pushing dataset")
    distiset = Distiset({"default": dataset})
    distiset.push_to_hub(
        repo_id=repo_id,
        private=private,
        include_script=False,
        token=oauth_token.token,
        create_pr=False,
    )
    push_pipeline_code_to_hub(pipeline_code, org_name, repo_name, oauth_token)
    progress(1.0, desc="Dataset pushed")
    return original_dataframe


def push_dataset(
    org_name: str,
    repo_name: str,
    system_prompt: str,
    num_turns: int = 1,
    num_rows: int = 10,
    private: bool = False,
    temperature: float = 0.9,
    pipeline_code: str = "",
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    dataframe = generate_dataset(
        system_prompt=system_prompt,
        num_turns=num_turns,
        num_rows=num_rows,
        temperature=temperature,
    )
    push_dataset_to_hub(
        dataframe, org_name, repo_name, oauth_token, private, pipeline_code
    )
    try:
        progress(0.1, desc="Setting up user and workspace")
        hf_user = HfApi().whoami(token=oauth_token.token)["name"]
        client = get_argilla_client()
        if client is None:
            return ""
        progress(0.5, desc="Creating dataset in Argilla")
        if "messages" in dataframe.columns:
            settings = rg.Settings(
                fields=[
                    rg.ChatField(
                        name="messages",
                        description="The messages in the conversation",
                        title="Messages",
                    ),
                ],
                questions=[
                    rg.RatingQuestion(
                        name="rating",
                        title="Rating",
                        description="The rating of the conversation",
                        values=list(range(1, 6)),
                    ),
                ],
                metadata=[
                    rg.IntegerMetadataProperty(
                        name="user_message_length", title="User Message Length"
                    ),
                    rg.IntegerMetadataProperty(
                        name="assistant_message_length",
                        title="Assistant Message Length",
                    ),
                ],
                vectors=[
                    rg.VectorField(
                        name="messages_embeddings",
                        dimensions=get_sentence_embedding_dimensions(),
                    )
                ],
                guidelines="Please review the conversation and provide a score for the assistant's response.",
            )

            dataframe["user_message_length"] = dataframe["messages"].apply(
                lambda x: sum([len(y["content"]) for y in x if y["role"] == "user"])
            )
            dataframe["assistant_message_length"] = dataframe["messages"].apply(
                lambda x: sum(
                    [len(y["content"]) for y in x if y["role"] == "assistant"]
                )
            )
            dataframe["messages_embeddings"] = get_embeddings(
                dataframe["messages"].apply(
                    lambda x: " ".join([y["content"] for y in x])
                )
            )
        else:
            settings = rg.Settings(
                fields=[
                    rg.TextField(
                        name="system_prompt",
                        title="System Prompt",
                        description="The system prompt used for the conversation",
                        required=False,
                    ),
                    rg.TextField(
                        name="prompt",
                        title="Prompt",
                        description="The prompt used for the conversation",
                    ),
                    rg.TextField(
                        name="completion",
                        title="Completion",
                        description="The completion from the assistant",
                    ),
                ],
                questions=[
                    rg.RatingQuestion(
                        name="rating",
                        title="Rating",
                        description="The rating of the conversation",
                        values=list(range(1, 6)),
                    ),
                ],
                metadata=[
                    rg.IntegerMetadataProperty(
                        name="prompt_length", title="Prompt Length"
                    ),
                    rg.IntegerMetadataProperty(
                        name="completion_length", title="Completion Length"
                    ),
                ],
                vectors=[
                    rg.VectorField(
                        name="prompt_embeddings",
                        dimensions=get_sentence_embedding_dimensions(),
                    )
                ],
                guidelines="Please review the conversation and correct the prompt and completion where needed.",
            )
            dataframe["prompt_length"] = dataframe["prompt"].apply(len)
            dataframe["completion_length"] = dataframe["completion"].apply(len)
            dataframe["prompt_embeddings"] = get_embeddings(dataframe["prompt"])

        rg_dataset = client.datasets(name=repo_name, workspace=hf_user)
        if rg_dataset is None:
            rg_dataset = rg.Dataset(
                name=repo_name,
                workspace=hf_user,
                settings=settings,
                client=client,
            )
            rg_dataset = rg_dataset.create()
        progress(0.7, desc="Pushing dataset to Argilla")
        hf_dataset = Dataset.from_pandas(dataframe)
        rg_dataset.records.log(records=hf_dataset)
        progress(1.0, desc="Dataset pushed to Argilla")
    except Exception as e:
        raise gr.Error(f"Error pushing dataset to Argilla: {e}")
    return ""


def show_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=True)}


def hide_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=False)}


######################
# Gradio UI
######################


with gr.Blocks() as app:
    with gr.Column() as main_ui:
        if not SFT_AVAILABLE:
            gr.Markdown(
                value="\n".join(
                    [
                        "## Supervised Fine-Tuning not available",
                        "",
                        f"This tool relies on the [Magpie](https://arxiv.org/abs/2406.08464) prequery template, which is not implemented for the {MODEL} with {BASE_URL}.",
                        "Use Llama3 or Qwen2 models with Hugging Face Inference Endpoints.",
                    ]
                )
            )
        else:
            gr.Markdown(value="## 1. Describe the dataset you want")
            with gr.Row():
                with gr.Column(scale=2):
                    dataset_description = gr.Textbox(
                        label="Dataset description",
                        placeholder="Give a precise description of your desired dataset.",
                    )
                    with gr.Row():
                        clear_btn_part = gr.Button(
                            "Clear",
                            variant="secondary",
                        )
                        load_btn = gr.Button(
                            "Create",
                            variant="primary",
                        )
                with gr.Column(scale=3):
                    examples = gr.Examples(
                        examples=DEFAULT_DATASET_DESCRIPTIONS,
                        inputs=[dataset_description],
                        cache_examples=False,
                        label="Examples",
                    )

            gr.HTML(value="<hr>")
            gr.Markdown(value="## 2. Configure your dataset")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    system_prompt = gr.Textbox(
                        label="System prompt",
                        placeholder="You are a helpful assistant.",
                    )
                    num_turns = gr.Number(
                        value=1,
                        label="Number of turns in the conversation",
                        minimum=1,
                        maximum=4,
                        step=1,
                        interactive=True,
                        info="Choose between 1 (single turn with 'instruction-response' columns) and 2-4 (multi-turn conversation with a 'messages' column).",
                    )
                    with gr.Row():
                        clear_btn_full = gr.Button(
                            "Clear",
                            variant="secondary",
                        )
                        btn_apply_to_sample_dataset = gr.Button(
                            "Save", variant="primary"
                        )
                with gr.Column(scale=3):
                    dataframe = _get_dataframe()

            gr.HTML(value="<hr>")
            gr.Markdown(value="## 3. Generate your dataset")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    org_name = get_org_dropdown()
                    repo_name = gr.Textbox(
                        label="Repo name",
                        placeholder="dataset_name",
                        value=f"my-distiset-{str(uuid.uuid4())[:8]}",
                        interactive=True,
                    )
                    num_rows = gr.Number(
                        label="Number of rows",
                        value=10,
                        interactive=True,
                        scale=1,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                    )
                    private = gr.Checkbox(
                        label="Private dataset",
                        value=False,
                        interactive=True,
                        scale=1,
                    )
                    btn_push_to_hub = gr.Button(
                        "Push to Hub", variant="primary", scale=2
                    )
                with gr.Column(scale=3):
                    success_message = gr.Markdown(
                        visible=True,
                        min_height=100,  # don't remove this otherwise progress is not visible
                    )
                    with gr.Accordion(
                        "Customize your pipeline with distilabel",
                        open=False,
                        visible=False,
                    ) as pipeline_code_ui:
                        code = generate_pipeline_code(
                            system_prompt=system_prompt.value,
                            num_turns=num_turns.value,
                            num_rows=num_rows.value,
                        )
                        pipeline_code = gr.Code(
                            value=code,
                            language="python",
                            label="Distilabel Pipeline Code",
                        )

            load_btn.click(
                fn=generate_system_prompt,
                inputs=[dataset_description],
                outputs=[system_prompt],
                show_progress=True,
            ).then(
                fn=generate_sample_dataset,
                inputs=[system_prompt, num_turns],
                outputs=[dataframe],
                show_progress=True,
            )

            btn_apply_to_sample_dataset.click(
                fn=generate_sample_dataset,
                inputs=[system_prompt, num_turns],
                outputs=[dataframe],
                show_progress=True,
            )

            btn_push_to_hub.click(
                fn=validate_argilla_user_workspace_dataset,
                inputs=[repo_name],
                outputs=[success_message],
                show_progress=True,
            ).then(
                fn=validate_push_to_hub,
                inputs=[org_name, repo_name],
                outputs=[success_message],
                show_progress=True,
            ).success(
                fn=hide_success_message,
                outputs=[success_message],
                show_progress=True,
            ).success(
                fn=hide_pipeline_code_visibility,
                inputs=[],
                outputs=[pipeline_code_ui],
                show_progress=True,
            ).success(
                fn=push_dataset,
                inputs=[
                    org_name,
                    repo_name,
                    system_prompt,
                    num_turns,
                    num_rows,
                    private,
                    temperature,
                    pipeline_code,
                ],
                outputs=[success_message],
                show_progress=True,
            ).success(
                fn=show_success_message,
                inputs=[org_name, repo_name],
                outputs=[success_message],
            ).success(
                fn=generate_pipeline_code,
                inputs=[system_prompt, num_turns, num_rows],
                outputs=[pipeline_code],
            ).success(
                fn=show_pipeline_code_visibility,
                inputs=[],
                outputs=[pipeline_code_ui],
            )
            gr.on(
                triggers=[clear_btn_part.click, clear_btn_full.click],
                fn=lambda _: ("", "", 1, _get_dataframe()),
                inputs=[dataframe],
                outputs=[system_prompt, num_turns, dataframe],
            )
            app.load(fn=get_org_dropdown, outputs=[org_name])
        app.load(fn=get_random_repo_name, outputs=[repo_name])
        app.load(fn=swap_visibility, outputs=main_ui)
