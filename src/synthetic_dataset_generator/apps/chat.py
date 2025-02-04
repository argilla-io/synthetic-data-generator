import ast
import json
import random
import uuid
from typing import Dict, List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import Dataset
from distilabel.distiset import Distiset
from gradio.oauth import OAuthToken
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from huggingface_hub import HfApi

from synthetic_dataset_generator.apps.base import (
    combine_datasets,
    hide_success_message,
    load_dataset_from_hub,
    preprocess_input_data,
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
    MODEL_COMPLETION,
    SFT_AVAILABLE,
)
from synthetic_dataset_generator.pipelines.base import get_rewritten_prompts
from synthetic_dataset_generator.pipelines.chat import (
    DEFAULT_DATASET_DESCRIPTIONS,
    generate_pipeline_code,
    get_follow_up_generator,
    get_magpie_generator,
    get_prompt_generator,
    get_response_generator,
    get_sentence_pair_generator,
)
from synthetic_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from synthetic_dataset_generator.utils import (
    column_to_list,
    get_argilla_client,
    get_org_dropdown,
    get_random_repo_name,
    swap_visibility,
)


def _get_dataframe():
    return gr.Dataframe(
        headers=["prompt", "completion"],
        wrap=True,
        interactive=False,
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


def generate_system_prompt(dataset_description: str, progress=gr.Progress()):
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


def load_dataset_file(
    repo_id: str,
    file_paths: list[str],
    input_type: str,
    num_rows: int = 10,
    token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
):
    progress(0.1, desc="Loading the source data")
    if input_type == "dataset-input":
        return load_dataset_from_hub(repo_id=repo_id, num_rows=num_rows, token=token)
    else:
        return preprocess_input_data(file_paths=file_paths, num_rows=num_rows)


def generate_sample_dataset(
    repo_id: str,
    file_paths: list[str],
    input_type: str,
    system_prompt: str,
    document_column: str,
    num_turns: int,
    num_rows: int,
    oauth_token: Union[OAuthToken, None],
    progress=gr.Progress(),
):
    if input_type == "prompt-input":
        dataframe = pd.DataFrame(columns=["prompt", "completion"])
    else:
        dataframe, _ = load_dataset_file(
            repo_id=repo_id,
            file_paths=file_paths,
            input_type=input_type,
            num_rows=num_rows,
            token=oauth_token,
        )
    progress(0.5, desc="Generating sample dataset")
    dataframe = generate_dataset(
        input_type=input_type,
        dataframe=dataframe,
        system_prompt=system_prompt,
        document_column=document_column,
        num_turns=num_turns,
        num_rows=num_rows,
        is_sample=True,
    )
    progress(1.0, desc="Sample dataset generated")
    return dataframe


def generate_dataset_from_prompt(
    system_prompt: str,
    num_turns: int = 1,
    num_rows: int = 10,
    temperature: float = 0.9,
    temperature_completion: Union[float, None] = None,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pd.DataFrame:
    num_rows = test_max_num_rows(num_rows)
    progress(0.0, desc="(1/2) Generating instructions")
    magpie_generator = get_magpie_generator(num_turns, temperature, is_sample)
    response_generator = get_response_generator(
        system_prompt=system_prompt,
        num_turns=num_turns,
        temperature=temperature or temperature_completion,
        is_sample=is_sample,
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


def generate_dataset_from_seed(
    dataframe: pd.DataFrame,
    document_column: str,
    num_turns: int = 1,
    num_rows: int = 10,
    temperature: float = 0.9,
    temperature_completion: Union[float, None] = None,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pd.DataFrame:
    num_rows = test_max_num_rows(num_rows)
    progress(0.0, desc="Initializing dataset generation")
    document_data = column_to_list(dataframe, document_column)
    if len(document_data) < num_rows:
        document_data += random.choices(document_data, k=num_rows - len(document_data))
    instruction_generator = get_sentence_pair_generator(
        temperature=temperature, is_sample=is_sample
    )
    response_generator = get_response_generator(
        system_prompt=None,
        num_turns=1,
        temperature=temperature or temperature_completion,
        is_sample=is_sample,
    )
    follow_up_generator_instruction = get_follow_up_generator(
        type="instruction", temperature=temperature, is_sample=is_sample
    )
    follow_up_generator_response = get_follow_up_generator(
        type="response",
        temperature=temperature or temperature_completion,
        is_sample=is_sample,
    )
    steps = 2 * num_turns
    total_steps: int = num_rows * steps
    step_progress = round(1 / steps, 2)
    batch_size = DEFAULT_BATCH_SIZE

    # create instructions
    n_processed = 0
    instruction_results = []
    while n_processed < num_rows:
        progress(
            step_progress * n_processed / num_rows,
            total=total_steps,
            desc="Generating questions",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        batch = [
            {"anchor": document}
            for document in document_data[n_processed : n_processed + batch_size]
        ]
        questions = list(instruction_generator.process(inputs=batch))
        instruction_results.extend(questions[0])
        n_processed += batch_size
    for result in instruction_results:
        result["instruction"] = result["positive"]
        result["prompt"] = result.pop("positive")

    progress(step_progress, desc="Generating instructions")

    # generate responses
    n_processed = 0
    response_results = []
    while n_processed < num_rows:
        progress(
            step_progress + step_progress * n_processed / num_rows,
            total=total_steps,
            desc="Generating responses",
        )
        batch = instruction_results[n_processed : n_processed + batch_size]
        responses = list(response_generator.process(inputs=batch))
        response_results.extend(responses[0])
        n_processed += batch_size
    for result in response_results:
        result["completion"] = result.pop("generation")

    # generate follow-ups
    if num_turns > 1:
        n_processed = 0
        final_conversations = []

        while n_processed < num_rows:
            progress(
                step_progress + step_progress * n_processed / num_rows,
                total=total_steps,
                desc="Generating follow-ups",
            )
            batch = response_results[n_processed : n_processed + batch_size]
            conversations_batch = [
                {
                    "messages": [
                        {"role": "user", "content": result["prompt"]},
                        {"role": "assistant", "content": result["completion"]},
                    ]
                }
                for result in batch
            ]

            for _ in range(num_turns - 1):
                follow_up_instructions = list(
                    follow_up_generator_instruction.process(inputs=conversations_batch)
                )
                for conv, follow_up in zip(conversations_batch, follow_up_instructions[0]):
                    conv["messages"].append(
                        {"role": "user", "content": follow_up["generation"]}
                    )

                follow_up_responses = list(
                    follow_up_generator_response.process(inputs=conversations_batch)
                )
                for conv, follow_up in zip(conversations_batch, follow_up_responses[0]):
                    conv["messages"].append(
                        {"role": "assistant", "content": follow_up["generation"]}
                    )

            final_conversations.extend(
                [{"messages": conv["messages"]} for conv in conversations_batch]
            )
            n_processed += batch_size

    # create distiset
    distiset_results = []
    if num_turns == 1:
        for result in response_results:
            record = {}
            for relevant_keys in ["prompt", "completion"]:
                if relevant_keys in result:
                    record[relevant_keys] = result[relevant_keys]
            distiset_results.append(record)
        dataframe = pd.DataFrame(distiset_results)
    else:
        distiset_results = final_conversations
        dataframe = pd.DataFrame(distiset_results)
        dataframe["messages"] = dataframe["messages"].apply(lambda x: json.dumps(x))

    progress(1.0, desc="Dataset generation completed")
    return dataframe


def generate_dataset(
    input_type: str,
    dataframe: pd.DataFrame,
    system_prompt: str,
    document_column: str,
    num_turns: int = 1,
    num_rows: int = 10,
    temperature: float = 0.9,
    temperature_completion: Union[float, None] = None,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pd.DataFrame:
    if input_type == "prompt-input":
        dataframe = generate_dataset_from_prompt(
            system_prompt=system_prompt,
            num_turns=num_turns,
            num_rows=num_rows,
            temperature=temperature,
            temperature_completion=temperature_completion,
            is_sample=is_sample,
        )
    else:
        dataframe = generate_dataset_from_seed(
            dataframe=dataframe,
            document_column=document_column,
            num_turns=num_turns,
            num_rows=num_rows,
            temperature=temperature,
            temperature_completion=temperature_completion,
            is_sample=is_sample,
        )
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
    private: bool,
    original_repo_id: str,
    file_paths: list[str],
    input_type: str,
    system_prompt: str,
    document_column: str,
    num_turns: int = 1,
    num_rows: int = 10,
    temperature: float = 0.9,
    temperature_completion: Union[float, None] = None,
    pipeline_code: str = "",
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    if input_type == "prompt-input":
        dataframe = _get_dataframe()
    else:
        dataframe, _ = load_dataset_file(
            repo_id=original_repo_id,
            file_paths=file_paths,
            input_type=input_type,
            num_rows=num_rows,
            token=oauth_token,
        )
    progress(0.5, desc="Generating dataset")
    dataframe = generate_dataset(
        input_type=input_type,
        dataframe=dataframe,
        system_prompt=system_prompt,
        document_column=document_column,
        num_turns=num_turns,
        num_rows=num_rows,
        temperature=temperature,
        temperature_completion=temperature_completion
    )
    push_dataset_to_hub(
        dataframe=dataframe,
        org_name=org_name,
        repo_name=repo_name,
        oauth_token=oauth_token,
        private=private,
        pipeline_code=pipeline_code,
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


def show_system_prompt_visibility():
    return {system_prompt: gr.Textbox(visible=True)}


def hide_system_prompt_visibility():
    return {system_prompt: gr.Textbox(visible=False)}


def show_document_column_visibility():
    return {document_column: gr.Dropdown(visible=True)}


def hide_document_column_visibility():
    return {
        document_column: gr.Dropdown(
            choices=["Load your data first in step 1."],
            value="Load your data first in step 1.",
            visible=False,
        )
    }


def show_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=True)}


def hide_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=False)}


def show_temperature_completion():
    if MODEL != MODEL_COMPLETION:
        return {temperature_completion: gr.Slider(value=0.9, visible=True)}


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
            gr.Markdown("## 1. Select your input")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    input_type = gr.Dropdown(
                        label="Input type",
                        choices=["prompt-input", "dataset-input", "file-input"],
                        value="prompt-input",
                        multiselect=False,
                        visible=False,
                    )
                    with gr.Tab("Generate from prompt") as tab_prompt_input:
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                dataset_description = gr.Textbox(
                                    label="Dataset description",
                                    placeholder="Give a precise description of your desired dataset.",
                                )
                                with gr.Row():
                                    clear_prompt_btn_part = gr.Button(
                                        "Clear", variant="secondary"
                                    )
                                    load_prompt_btn = gr.Button(
                                        "Create", variant="primary"
                                    )
                            with gr.Column(scale=3):
                                examples = gr.Examples(
                                    examples=DEFAULT_DATASET_DESCRIPTIONS,
                                    inputs=[dataset_description],
                                    cache_examples=False,
                                    label="Examples",
                                )
                    with gr.Tab("Load from Hub") as tab_dataset_input:
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                search_in = HuggingfaceHubSearch(
                                    label="Search",
                                    placeholder="Search for a dataset",
                                    search_type="dataset",
                                    sumbit_on_select=True,
                                )
                                with gr.Row():
                                    clear_dataset_btn_part = gr.Button(
                                        "Clear", variant="secondary"
                                    )
                                    load_dataset_btn = gr.Button(
                                        "Load", variant="primary"
                                    )
                            with gr.Column(scale=3):
                                examples = gr.Examples(
                                    examples=[
                                        "charris/wikipedia_sample",
                                        "plaguss/argilla_sdk_docs_raw_unstructured",
                                        "BeIR/hotpotqa-generated-queries",
                                    ],
                                    label="Example datasets",
                                    fn=lambda x: x,
                                    inputs=[search_in],
                                    run_on_click=True,
                                )
                                search_out = gr.HTML(
                                    label="Dataset preview", visible=False
                                )
                    with gr.Tab("Load your file") as tab_file_input:
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2):
                                file_in = gr.File(
                                    label="Upload your file. Supported formats: .md, .txt, .docx, .pdf",
                                    file_count="multiple",
                                    file_types=[".md", ".txt", ".docx", ".pdf"],
                                )
                                with gr.Row():
                                    clear_file_btn_part = gr.Button(
                                        "Clear", variant="secondary"
                                    )
                                    load_file_btn = gr.Button("Load", variant="primary")
                            with gr.Column(scale=3):
                                file_out = gr.HTML(
                                    label="Dataset preview", visible=False
                                )

            gr.HTML(value="<hr>")
            gr.Markdown(value="## 2. Configure your dataset")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    system_prompt = gr.Textbox(
                        label="System prompt",
                        placeholder="You are a helpful assistant.",
                    )
                    document_column = gr.Dropdown(
                        label="Document Column",
                        info="Select the document column to generate the RAG dataset",
                        choices=["Load your data first in step 1."],
                        value="Load your data first in step 1.",
                        interactive=False,
                        multiselect=False,
                        allow_custom_value=False,
                        visible=False,
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
                        maximum=1.5,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                    )
                    temperature_completion = gr.Slider(
                        label="Temperature for completion",
                        minimum=0.1,
                        maximum=1.5,
                        value=None,
                        step=0.1,
                        interactive=True,
                        visible=False,
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
                            repo_id=search_in.value,
                            input_type=input_type.value,
                            system_prompt=system_prompt.value,
                            document_column=document_column.value,
                            num_turns=num_turns.value,
                            num_rows=num_rows.value,
                        )
                        pipeline_code = gr.Code(
                            value=code,
                            language="python",
                            label="Distilabel Pipeline Code",
                        )

    tab_prompt_input.select(
        fn=lambda: "prompt-input",
        inputs=[],
        outputs=[input_type],
    ).then(fn=show_system_prompt_visibility, inputs=[], outputs=[system_prompt]).then(
        fn=hide_document_column_visibility, inputs=[], outputs=[document_column]
    )

    tab_dataset_input.select(
        fn=lambda: "dataset-input",
        inputs=[],
        outputs=[input_type],
    ).then(fn=hide_system_prompt_visibility, inputs=[], outputs=[system_prompt]).then(
        fn=show_document_column_visibility, inputs=[], outputs=[document_column]
    )

    tab_file_input.select(
        fn=lambda: "file-input",
        inputs=[],
        outputs=[input_type],
    ).then(fn=hide_system_prompt_visibility, inputs=[], outputs=[system_prompt]).then(
        fn=show_document_column_visibility, inputs=[], outputs=[document_column]
    )

    search_in.submit(
        fn=lambda df: pd.DataFrame(columns=df.columns),
        inputs=[dataframe],
        outputs=[dataframe],
    )

    load_prompt_btn.click(
        fn=generate_system_prompt,
        inputs=[dataset_description],
        outputs=[system_prompt],
    ).success(
        fn=generate_sample_dataset,
        inputs=[
            search_in,
            file_in,
            input_type,
            system_prompt,
            document_column,
            num_turns,
            num_rows,
        ],
        outputs=dataframe,
    )

    gr.on(
        triggers=[load_dataset_btn.click, load_file_btn.click],
        fn=load_dataset_file,
        inputs=[search_in, file_in, input_type],
        outputs=[dataframe, document_column],
    )

    btn_apply_to_sample_dataset.click(
        fn=generate_sample_dataset,
        inputs=[
            search_in,
            file_in,
            input_type,
            system_prompt,
            document_column,
            num_turns,
            num_rows,
        ],
        outputs=dataframe,
    )

    btn_push_to_hub.click(
        fn=validate_argilla_user_workspace_dataset,
        inputs=[repo_name],
        outputs=[success_message],
    ).then(
        fn=validate_push_to_hub,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    ).success(
        fn=hide_success_message,
        outputs=[success_message],
    ).success(
        fn=hide_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    ).success(
        fn=push_dataset,
        inputs=[
            org_name,
            repo_name,
            private,
            search_in,
            file_in,
            input_type,
            system_prompt,
            document_column,
            num_turns,
            num_rows,
            temperature,
            temperature_completion,
            pipeline_code,
        ],
        outputs=[success_message],
    ).success(
        fn=show_success_message,
        inputs=[org_name, repo_name],
        outputs=[success_message],
    ).success(
        fn=generate_pipeline_code,
        inputs=[
            search_in,
            input_type,
            system_prompt,
            document_column,
            num_turns,
            num_rows,
        ],
        outputs=[pipeline_code],
    ).success(
        fn=show_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    )

    clear_dataset_btn_part.click(fn=lambda: "", inputs=[], outputs=[search_in])
    clear_file_btn_part.click(fn=lambda: None, inputs=[], outputs=[file_in])
    clear_prompt_btn_part.click(fn=lambda: "", inputs=[], outputs=[dataset_description])
    clear_btn_full.click(
        fn=lambda df: ("", "", [], _get_dataframe()),
        inputs=[dataframe],
        outputs=[system_prompt, document_column, num_turns, dataframe],
    )
    app.load(fn=swap_visibility, outputs=main_ui)
    app.load(fn=get_org_dropdown, outputs=[org_name])
    app.load(fn=get_random_repo_name, outputs=[repo_name])
    app.load(fn=show_temperature_completion, outputs=[temperature_completion])
