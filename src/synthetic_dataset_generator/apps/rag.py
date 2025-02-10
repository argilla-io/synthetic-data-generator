import os
import random
import uuid
from typing import Union

import argilla as rg
import gradio as gr
import nltk
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
from synthetic_dataset_generator.constants import DEFAULT_BATCH_SIZE, MODEL, MODEL_COMPLETION, SAVE_LOCAL_DIR
from synthetic_dataset_generator.pipelines.base import get_rewritten_prompts
from synthetic_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from synthetic_dataset_generator.pipelines.rag import (
    DEFAULT_DATASET_DESCRIPTIONS,
    generate_pipeline_code,
    get_chunks_generator,
    get_prompt_generator,
    get_response_generator,
    get_sentence_pair_generator,
)
from synthetic_dataset_generator.utils import (
    column_to_list,
    get_argilla_client,
    get_org_dropdown,
    get_random_repo_name,
    swap_visibility,
)

os.makedirs("./nltk_data", exist_ok=True)
nltk.data.path.append("./nltk_data")
nltk.download("punkt_tab", download_dir="./nltk_data")
nltk.download("averaged_perceptron_tagger_eng", download_dir="./nltk_data")


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
    retrieval_reranking: list[str],
    num_rows: str,
    oauth_token: Union[OAuthToken, None],
    progress=gr.Progress(),
):
    retrieval = "Retrieval" in retrieval_reranking
    reranking = "Reranking" in retrieval_reranking

    if input_type == "prompt-input":
        dataframe = pd.DataFrame(columns=["context", "question", "response"])
    else:
        dataframe, _ = load_dataset_file(
            repo_id=repo_id,
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
        retrieval=retrieval,
        reranking=reranking,
        num_rows=10,
        is_sample=True,
    )
    progress(1.0, desc="Sample dataset generated")
    return dataframe


def generate_dataset(
    input_type: str,
    dataframe: pd.DataFrame,
    system_prompt: str,
    document_column: str,
    retrieval: bool = False,
    reranking: bool = False,
    num_rows: int = 10,
    temperature: float = 0.7,
    temperature_completion: Union[float, None] = None,
    is_sample: bool = False,
    progress=gr.Progress(),
):
    num_rows = test_max_num_rows(num_rows)
    progress(0.0, desc="Initializing dataset generation")
    if input_type == "prompt-input":
        chunk_generator = get_chunks_generator(
            temperature=temperature, is_sample=is_sample
        )
    else:
        document_data = column_to_list(dataframe, document_column)
        if len(document_data) < num_rows:
            document_data += random.choices(
                document_data, k=num_rows - len(document_data)
            )

    retrieval_generator = get_sentence_pair_generator(
        action="query",
        triplet=True if retrieval else False,
        temperature=temperature,
        is_sample=is_sample,
    )
    response_generator = get_response_generator(
        temperature = temperature_completion or temperature , is_sample=is_sample
    )
    if reranking:
        reranking_generator = get_sentence_pair_generator(
            action="semantically-similar",
            triplet=True,
            temperature=temperature,
            is_sample=is_sample,
        )
    steps = 2 + sum([1 if reranking else 0, 1 if input_type == "prompt-type" else 0])
    total_steps: int = num_rows * steps
    step_progress = round(1 / steps, 2)
    batch_size = DEFAULT_BATCH_SIZE

    # generate chunks
    if input_type == "prompt-input":
        n_processed = 0
        chunk_results = []
        rewritten_system_prompts = get_rewritten_prompts(system_prompt, num_rows)
        while n_processed < num_rows:
            progress(
                step_progress * n_processed / num_rows,
                total=total_steps,
                desc="Generating chunks",
            )
            remaining_rows = num_rows - n_processed
            batch_size = min(batch_size, remaining_rows)
            inputs = [
                {"task": random.choice(rewritten_system_prompts)}
                for _ in range(batch_size)
            ]
            chunks = list(chunk_generator.process(inputs=inputs))
            chunk_results.extend(chunks[0])
            n_processed += batch_size
            random.seed(a=random.randint(0, 2**32 - 1))
        document_data = [chunk["generation"] for chunk in chunk_results]
        progress(step_progress, desc="Generating chunks")

    # generate questions
    n_processed = 0
    retrieval_results = []
    while n_processed < num_rows:
        progress(
            step_progress * n_processed / num_rows,
            total=total_steps,
            desc="Generating questions",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        inputs = [
            {"anchor": document}
            for document in document_data[n_processed : n_processed + batch_size]
        ]
        questions = list(retrieval_generator.process(inputs=inputs))
        retrieval_results.extend(questions[0])
        n_processed += batch_size
    for result in retrieval_results:
        result["context"] = result["anchor"]
        if retrieval:
            result["question"] = result["positive"]
            result["positive_retrieval"] = result.pop("positive")
            result["negative_retrieval"] = result.pop("negative")
        else:
            result["question"] = result.pop("positive")

    progress(step_progress, desc="Generating questions")

    # generate responses
    n_processed = 0
    response_results = []
    while n_processed < num_rows:
        progress(
            step_progress + step_progress * n_processed / num_rows,
            total=total_steps,
            desc="Generating responses",
        )
        batch = retrieval_results[n_processed : n_processed + batch_size]
        responses = list(response_generator.process(inputs=batch))
        response_results.extend(responses[0])
        n_processed += batch_size
    for result in response_results:
        result["response"] = result["generation"]
    progress(step_progress, desc="Generating responses")

    # generate reranking
    if reranking:
        n_processed = 0
        reranking_results = []
        while n_processed < num_rows:
            progress(
                step_progress * n_processed / num_rows,
                total=total_steps,
                desc="Generating reranking data",
            )
            batch = response_results[n_processed : n_processed + batch_size]
            batch = list(reranking_generator.process(inputs=batch))
            reranking_results.extend(batch[0])
            n_processed += batch_size
        for result in reranking_results:
            result["positive_reranking"] = result.pop("positive")
            result["negative_reranking"] = result.pop("negative")
    progress(
        1,
        total=total_steps,
        desc="Creating dataset",
    )

    # create distiset
    distiset_results = []
    source_results = reranking_results if reranking else response_results
    base_keys = ["context", "question", "response"]
    retrieval_keys = ["positive_retrieval", "negative_retrieval"] if retrieval else []
    reranking_keys = ["positive_reranking", "negative_reranking"] if reranking else []
    relevant_keys = base_keys + retrieval_keys + reranking_keys

    for result in source_results:
        record = {key: result.get(key) for key in relevant_keys if key in result}
        distiset_results.append(record)

    dataframe = pd.DataFrame(distiset_results)

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
    progress(0.5, desc="Creating dataset")
    dataset = Dataset.from_pandas(dataframe)
    dataset = combine_datasets(repo_id, dataset, oauth_token)
    distiset = Distiset({"default": dataset})
    progress(0.9, desc="Pushing dataset")
    distiset.push_to_hub(
        repo_id=repo_id,
        private=private,
        include_script=False,
        token=oauth_token.token,
        create_pr=False,
    )
    push_pipeline_code_to_hub(pipeline_code, org_name, repo_name, oauth_token)
    progress(1.0, desc="Dataset pushed")
    return dataframe


def push_dataset(
    org_name: str,
    repo_name: str,
    private: bool,
    original_repo_id: str,
    file_paths: list[str],
    input_type: str,
    system_prompt: str,
    document_column: str,
    retrieval_reranking: list[str],
    num_rows: int,
    temperature: float,
    temperature_completion: float,
    pipeline_code: str,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    retrieval = "Retrieval" in retrieval_reranking
    reranking = "Reranking" in retrieval_reranking

    if input_type == "prompt-input":
        dataframe = pd.DataFrame(columns=["context", "question", "response"])
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
        retrieval=retrieval,
        reranking=reranking,
        num_rows=num_rows,
        temperature=temperature,
        temperature_completion=temperature_completion,
        is_sample=True,
    )
    push_dataset_to_hub(
        dataframe, org_name, repo_name, oauth_token, private, pipeline_code
    )
    dataframe = dataframe[
        dataframe.applymap(lambda x: str(x).strip() if pd.notna(x) else x).apply(
            lambda row: row.notna().all() and (row != "").all(), axis=1
        )
    ]
    try:
        progress(0.1, desc="Setting up user and workspace")
        hf_user = HfApi().whoami(token=oauth_token.token)["name"]
        client = get_argilla_client()
        if client is None:
            return ""

        progress(0.5, desc="Creating dataset in Argilla")
        fields = [
            rg.TextField(
                name="context",
                title="Context",
                description="Context for the generation",
            ),
            rg.ChatField(
                name="chat",
                title="Chat",
                description="User and assistant conversation based on the context",
            ),
        ]
        for item in ["positive", "negative"]:
            if retrieval:
                fields.append(
                    rg.TextField(
                        name=f"{item}_retrieval",
                        title=f"{item.capitalize()} retrieval",
                        description=f"The {item} query for retrieval",
                    )
                )
            if reranking:
                fields.append(
                    rg.TextField(
                        name=f"{item}_reranking",
                        title=f"{item.capitalize()} reranking",
                        description=f"The {item} query for reranking",
                    )
                )

        questions = [
            rg.LabelQuestion(
                name="relevant",
                title="Are the question and response relevant to the given context?",
                labels=["yes", "no"],
            ),
            rg.LabelQuestion(
                name="is_response_correct",
                title="Is the response correct?",
                labels=["yes", "no"],
            ),
        ]
        for item in ["positive", "negative"]:
            if retrieval:
                questions.append(
                    rg.LabelQuestion(
                        name=f"is_{item}_retrieval_relevant",
                        title=f"Is the {item} retrieval relevant?",
                        labels=["yes", "no"],
                        required=False,
                    )
                )
            if reranking:
                questions.append(
                    rg.LabelQuestion(
                        name=f"is_{item}_reranking_relevant",
                        title=f"Is the {item} reranking relevant?",
                        labels=["yes", "no"],
                        required=False,
                    )
                )
        metadata = [
            rg.IntegerMetadataProperty(
                name=f"{item}_length", title=f"{item.capitalize()} length"
            )
            for item in ["context", "question", "response"]
        ]

        vectors = [
            rg.VectorField(
                name=f"{item}_embeddings",
                dimensions=get_sentence_embedding_dimensions(),
            )
            for item in ["context", "question", "response"]
        ]
        settings = rg.Settings(
            fields=fields,
            questions=questions,
            metadata=metadata,
            vectors=vectors,
            guidelines="Please review the conversation and provide an evaluation.",
        )

        dataframe["chat"] = dataframe.apply(
            lambda row: [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["response"]},
            ],
            axis=1,
        )

        for item in ["context", "question", "response"]:
            dataframe[f"{item}_length"] = dataframe[item].apply(
                lambda x: len(x) if x is not None else 0
            )
            dataframe[f"{item}_embeddings"] = get_embeddings(
                dataframe[item].apply(lambda x: x if x is not None else "").to_list()
            )

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


def save_local(
    repo_id: str,
    file_paths: list[str],
    input_type: str,
    system_prompt: str,
    document_column: str,
    retrieval_reranking: list[str],
    num_rows: int,
    temperature: float,
    repo_name: str,
    temperature_completion: float,
) -> pd.DataFrame:
    retrieval = "Retrieval" in retrieval_reranking
    reranking = "Reranking" in retrieval_reranking

    if input_type == "prompt-input":
        dataframe = pd.DataFrame(columns=["context", "question", "response"])
    else:
        dataframe, _ = load_dataset_file(
            repo_id=repo_id,
            file_paths=file_paths,
            input_type=input_type,
            num_rows=num_rows,
        )
    dataframe = generate_dataset(
        input_type=input_type,
        dataframe=dataframe,
        system_prompt=system_prompt,
        document_column=document_column,
        retrieval=retrieval,
        reranking=reranking,
        num_rows=num_rows,
        temperature=temperature,
        temperature_completion=temperature_completion,
    )
    local_dataset = Dataset.from_pandas(dataframe)
    output_csv = os.path.join(SAVE_LOCAL_DIR, repo_name + ".csv")
    output_json = os.path.join(SAVE_LOCAL_DIR, repo_name + ".json")
    local_dataset.to_csv(output_csv, index=False)
    local_dataset.to_json(output_json, index=False)
    return output_csv, output_json


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


def show_save_local():
    return {
        btn_save_local: gr.Button(visible=True),
        csv_file: gr.File(visible=True),
        json_file: gr.File(visible=True)
    }


######################
# Gradio UI
######################


with gr.Blocks() as app:
    with gr.Column() as main_ui:
        gr.Markdown("## 1. Select your input")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                input_type = gr.Dropdown(
                    label="Input type",
                    choices=["dataset-input", "file-input", "prompt-input"],
                    value="dataset-input",
                    multiselect=False,
                    visible=False,
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
                                load_dataset_btn = gr.Button("Load", variant="primary")
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
                            search_out = gr.HTML(label="Dataset preview", visible=False)
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
                            file_out = gr.HTML(label="Dataset preview", visible=False)
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
                                load_prompt_btn = gr.Button("Create", variant="primary")
                        with gr.Column(scale=3):
                            examples = gr.Examples(
                                examples=DEFAULT_DATASET_DESCRIPTIONS,
                                inputs=[dataset_description],
                                cache_examples=False,
                                label="Examples",
                            )

        gr.HTML(value="<hr>")
        gr.Markdown(value="## 2. Configure your task")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                system_prompt = gr.Textbox(
                    label="System prompt",
                    placeholder="You are a helpful assistant.",
                    visible=False,
                )
                document_column = gr.Dropdown(
                    label="Document Column",
                    info="Select the document column to generate the RAG dataset",
                    choices=["Load your data first in step 1."],
                    value="Load your data first in step 1.",
                    interactive=False,
                    multiselect=False,
                    allow_custom_value=False,
                )
                retrieval_reranking = gr.CheckboxGroup(
                    choices=[("Retrieval", "Retrieval"), ("Reranking", "Reranking")],
                    type="value",
                    label="Data for RAG",
                    info="Indicate the additional data you want to generate for RAG.",
                )
                with gr.Row():
                    clear_btn_full = gr.Button("Clear", variant="secondary")
                    btn_apply_to_sample_dataset = gr.Button("Save", variant="primary")
            with gr.Column(scale=3):
                dataframe = gr.Dataframe(
                    headers=["context", "question", "response"],
                    wrap=True,
                    interactive=False,
                )

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
                    value=0.7,
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
                btn_save_local = gr.Button(
                    "Save locally", variant="primary", scale=2, visible=False
                )
                csv_file = gr.File(label="CSV", elem_classes="datasets", visible=False)
                json_file = gr.File(label="JSON", elem_classes="datasets", visible=False)
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
                        retrieval_reranking=retrieval_reranking.value,
                        num_rows=num_rows.value,
                    )
                    pipeline_code = gr.Code(
                        value=code,
                        language="python",
                        label="Distilabel Pipeline Code",
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

    tab_prompt_input.select(
        fn=lambda: "prompt-input",
        inputs=[],
        outputs=[input_type],
    ).then(fn=show_system_prompt_visibility, inputs=[], outputs=[system_prompt]).then(
        fn=hide_document_column_visibility, inputs=[], outputs=[document_column]
    )

    search_in.submit(
        fn=lambda df: pd.DataFrame(columns=df.columns),
        inputs=[dataframe],
        outputs=[dataframe],
    )

    gr.on(
        triggers=[load_dataset_btn.click, load_file_btn.click],
        fn=load_dataset_file,
        inputs=[search_in, file_in, input_type],
        outputs=[dataframe, document_column],
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
            retrieval_reranking,
            num_rows,
        ],
        outputs=dataframe,
    )

    btn_apply_to_sample_dataset.click(
        fn=generate_sample_dataset,
        inputs=[
            search_in,
            file_in,
            input_type,
            system_prompt,
            document_column,
            retrieval_reranking,
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
            retrieval_reranking,
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
            retrieval_reranking,
            num_rows,
        ],
        outputs=[pipeline_code],
    ).success(
        fn=show_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    )

    btn_save_local.click(
        save_local,
        inputs=[
            search_in,
            file_in,
            input_type,
            system_prompt,
            document_column,
            retrieval_reranking,
            num_rows,
            temperature,
            repo_name,
            temperature_completion,
        ],
        outputs=[csv_file, json_file]
    )

    clear_dataset_btn_part.click(fn=lambda: "", inputs=[], outputs=[search_in])
    clear_file_btn_part.click(fn=lambda: None, inputs=[], outputs=[file_in])
    clear_prompt_btn_part.click(fn=lambda: "", inputs=[], outputs=[dataset_description])
    clear_btn_full.click(
        fn=lambda df: ("", [], pd.DataFrame(columns=df.columns)),
        inputs=[dataframe],
        outputs=[document_column, retrieval_reranking, dataframe],
    )

    app.load(fn=swap_visibility, outputs=main_ui)
    app.load(fn=get_org_dropdown, outputs=[org_name])
    app.load(fn=get_random_repo_name, outputs=[repo_name])
    app.load(fn=show_temperature_completion, outputs=[temperature_completion])
    if SAVE_LOCAL_DIR is not None:
        app.load(fn=show_save_local, outputs=[btn_save_local, csv_file, json_file])
