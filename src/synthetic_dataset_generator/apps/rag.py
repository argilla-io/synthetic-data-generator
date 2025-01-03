import random
import uuid
from typing import Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from distilabel.distiset import Distiset
from gradio.oauth import OAuthToken
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from huggingface_hub import HfApi, repo_exists
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from synthetic_dataset_generator.apps.base import (
    combine_datasets,
    hide_success_message,
    push_pipeline_code_to_hub,
    show_success_message,
    test_max_num_rows,
    validate_argilla_user_workspace_dataset,
    validate_push_to_hub,
)
from synthetic_dataset_generator.constants import DEFAULT_BATCH_SIZE
from synthetic_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from synthetic_dataset_generator.pipelines.rag import (
    DEFAULT_DATASET_DESCRIPTIONS,
    get_chunks_generator,
    get_prompt_generator,
    generate_pipeline_code,
    get_sentence_pair_generator,
    get_response_generator,
)
from synthetic_dataset_generator.utils import (
    column_to_list,
    get_argilla_client,
    get_org_dropdown,
    swap_visibility,
)


def get_iframe(hub_repo_id: str) -> str:
    if not hub_repo_id:
        return ""

    if not repo_exists(repo_id=hub_repo_id, repo_type="dataset"):
        return ""

    url = f"https://huggingface.co/datasets/{hub_repo_id}/embed/viewer"
    iframe = f"""
    <iframe
        src="{url}"
        frameborder="0"
        width="100%"
        height="600px"
    ></iframe>
    """
    return iframe


def _get_valid_columns(dataframe: pd.DataFrame):
    doc_valid_columns = []

    for col in dataframe.columns:
        sample_val = dataframe[col].iloc[0]
        if isinstance(sample_val, str):
            doc_valid_columns.append(col)

    return doc_valid_columns


def _load_dataset_from_hub(
    repo_id: str, num_rows: int = 10, token: Union[OAuthToken, None] = None
):
    if not repo_id:
        raise gr.Error("Hub repo id is required")
    subsets = get_dataset_config_names(repo_id, token=token)
    splits = get_dataset_split_names(repo_id, subsets[0], token=token)
    ds = load_dataset(repo_id, subsets[0], split=splits[0], token=token, streaming=True)
    rows = []
    for idx, row in enumerate(ds):
        rows.append(row)
        if idx == num_rows:
            break
    ds = Dataset.from_list(rows)
    dataframe = ds.to_pandas()
    doc_valid_columns = _get_valid_columns(dataframe)
    col_doc = doc_valid_columns[0] if doc_valid_columns else ""
    return (
        dataframe,
        gr.Dropdown(
            choices=doc_valid_columns,
            label="Documents column",
            value=col_doc,
            interactive=(False if col_doc == "" else True),
        ),
    )


def _preprocess_input_data(file_paths, num_rows):
    data = {}
    total_chunks = 0
    for file_path in file_paths:
        partitioned_file = partition(filename=file_path)
        chunks = [str(chunk) for chunk in chunk_by_title(partitioned_file)]
        data[file_path] = chunks
        total_chunks += len(chunks)
        if total_chunks >= num_rows:
            break

    dataframe = pd.DataFrame.from_records(
        [(k, v) for k, values in data.items() for v in values],
        columns=["filename", "chunks"],
    )
    col_doc = "chunks"

    return (
        dataframe,
        gr.Dropdown(
            choices=["chucks"],
            label="Documents column",
            value=col_doc,
            interactive=(False if col_doc == "" else True),
        ),
    )


def generate_system_prompt(dataset_description, progress=gr.Progress()):
    progress(0.0, desc="Starting")
    progress(0.3, desc="Initializing")
    generate_description = get_prompt_generator()
    progress(0.7, desc="Generating")
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
):
    if input_type == "dataset-input":
        return _load_dataset_from_hub(repo_id, num_rows, token)
    else:
        return _preprocess_input_data(file_paths, num_rows)


def generate_dataset(
    input_type: str,
    dataframe: pd.DataFrame,
    system_prompt: str,
    document_column: str,
    hard_negative: bool = False,
    retrieval: bool = False,
    reranking: bool = False,
    num_rows: int = 10,
    temperature: float = 0.7,
    is_sample: bool = False,
    progress=gr.Progress(),
):
    num_rows = test_max_num_rows(num_rows)
    progress(0.0, desc="Generating questions")
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
        hard_negative=hard_negative,
        temperature=temperature,
        is_sample=is_sample,
    )
    response_generator = get_response_generator(
        temperature=temperature, is_sample=is_sample
    )
    if reranking:
        reranking_generator = get_sentence_pair_generator(
            action="semantically-similar",
            triplet=True,
            hard_negative=hard_negative,
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
        while n_processed < num_rows:
            progress(
                step_progress * n_processed / num_rows,
                total=total_steps,
                desc="Generating chunks",
            )
            remaining_rows = num_rows - n_processed
            batch_size = min(batch_size, remaining_rows)
            inputs = [{"task": system_prompt} for _ in range(batch_size)]
            chunks = list(chunk_generator.process(inputs=inputs))
            chunk_results.extend(chunks[0])
            n_processed += batch_size
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
    base_keys = ["context", "question", "response", "model_name"]
    retrieval_keys = ["positive_retrieval", "negative_retrieval"] if retrieval else []
    reranking_keys = ["positive_reranking", "negative_reranking"] if reranking else []
    relevant_keys = base_keys + retrieval_keys + reranking_keys

    for result in source_results:
        record = {key: result.get(key) for key in relevant_keys if key in result}
        distiset_results.append(record)

    dataframe = pd.DataFrame(distiset_results)

    progress(1.0, desc="Dataset generation completed")
    return dataframe


def generate_sample_dataset(
    repo_id: str,
    file_paths: list[str],
    input_type: str,
    system_prompt: str,
    document_column: str,
    hard_negative: bool,
    retrieval: bool,
    reranking: bool,
    num_rows: str,
    oauth_token: Union[OAuthToken, None],
):
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
    dataframe = generate_dataset(
        input_type=input_type,
        dataframe=dataframe,
        system_prompt=system_prompt,
        document_column=document_column,
        hard_negative=hard_negative,
        retrieval=retrieval,
        reranking=reranking,
        num_rows=1,  # TODO: 10
        is_sample=True,
    )
    return dataframe


def push_dataset_to_hub(
    dataframe: pd.DataFrame,
    org_name: str,
    repo_name: str,
    oauth_token: Union[gr.OAuthToken, None],
    private: bool,
    pipeline_code: str,
):
    repo_id = validate_push_to_hub(org_name, repo_name)
    dataset = Dataset.from_pandas(dataframe)
    dataset = combine_datasets(repo_id, dataset)
    distiset = Distiset({"default": dataset})
    distiset.push_to_hub(
        repo_id=repo_id,
        private=private,
        include_script=False,
        token=oauth_token.token,
        create_pr=False,
    )
    push_pipeline_code_to_hub(pipeline_code, org_name, repo_name, oauth_token)


def push_dataset(
    org_name: str,
    repo_name: str,
    private: bool,
    original_repo_id: str,
    file_paths: list[str],
    input_type: str,
    system_prompt: str,
    document_column: str,
    hard_negative: bool,
    retrieval: bool,
    reranking: bool,
    num_rows: int,
    temperature: float,
    pipeline_code: str,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    if input_type != "prompt-input":
        dataframe, _ = load_dataset_file(
            repo_id=original_repo_id,
            file_paths=file_paths,
            input_type=input_type,
            num_rows=num_rows,
            token=oauth_token,
        )
    dataframe = generate_dataset(
        input_type=input_type,
        dataframe=dataframe,
        system_prompt=system_prompt,
        document_column=document_column,
        hard_negative=hard_negative,
        retrieval=retrieval,
        reranking=reranking,
        num_rows=num_rows,
        temperature=temperature,
        is_sample=True,
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
            dataframe[f"{item}_length"] = dataframe[item].apply(len)
            dataframe[f"{item}_embeddings"] = get_embeddings(dataframe[item].to_list())

        progress(0.5, desc="Creating dataset")
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
    return {document_column: gr.Dropdown(visible=False)}


def show_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=True)}


def hide_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=False)}


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
                                file_count="multiple",
                                label="Upload your file",
                                file_types=[".md", ".txt"],
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
        with gr.Row(equal_height=True):
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
                    hard_negative = gr.Checkbox(
                        label="Hard Negative",
                        value=True,
                        interactive=True,
                        info="If checked, it will generate hard negative examples.",
                    )
                    retrieval = gr.Checkbox(
                        label="Retrieval",
                        value=False,
                        interactive=True,
                        info="If checked, it will generate data for retrieval.",
                    )
                    reranking = gr.Checkbox(
                        label="Reranking",
                        value=False,
                        interactive=True,
                        info="If checked, it will generate data for reranking.",
                    )
                    with gr.Row():
                        clear_btn_full = gr.Button("Clear", variant="secondary")
                        btn_apply_to_sample_dataset = gr.Button(
                            "Save", variant="primary"
                        )
                with gr.Column(scale=3):
                    dataframe = gr.Dataframe(
                        headers=["context", "query", "response"],
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
                    value=1,  # TODO: 10
                    interactive=True,
                    scale=1,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                )
                private = gr.Checkbox(
                    label="Private dataset",
                    value=False,
                    interactive=True,
                    scale=1,
                )
                btn_push_to_hub = gr.Button("Push to Hub", variant="primary", scale=2)
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
                        file_paths=file_in.value,
                        document_column=document_column.value,
                        retrieval=retrieval.value,
                        reranking=reranking.value,
                        hard_negative=hard_negative.value,
                        num_rows=num_rows.value,
                        temperature=temperature.value,
                        input_type=input_type.value,
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

    search_in.submit(fn=get_iframe, inputs=search_in, outputs=search_out).then(
        fn=lambda df: pd.DataFrame(columns=df.columns),
        inputs=[dataframe],
        outputs=[dataframe],
    )

    load_dataset_btn.click(
        fn=load_dataset_file,
        inputs=[search_in, file_in, input_type],
        outputs=[
            dataframe,
            document_column,
        ],
    )

    load_file_btn.click(
        fn=load_dataset_file,
        inputs=[search_in, file_in, input_type],
        outputs=[
            dataframe,
            document_column,
        ],
    )

    load_prompt_btn.click(
        fn=generate_system_prompt,
        inputs=[dataset_description],
        outputs=[system_prompt],
        show_progress=True,
    ).success(
        fn=generate_sample_dataset,
        inputs=[
            search_in,
            file_in,
            input_type,
            system_prompt,
            document_column,
            hard_negative,
            retrieval,
            reranking,
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
            hard_negative,
            retrieval,
            reranking,
            num_rows,
        ],
        outputs=dataframe,
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
            hard_negative,
            retrieval,
            reranking,
            num_rows,
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
        inputs=[
            search_in,
            file_in,
            input_type,
            document_column,
            hard_negative,
            retrieval,
            reranking,
            num_rows,
            temperature,
        ],
        outputs=[pipeline_code],
    ).success(
        fn=show_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    )

    clear_dataset_btn_part.click(fn=lambda x: "", inputs=[], outputs=[search_in])
    clear_file_btn_part.click(fn=lambda x: "", inputs=[], outputs=[file_in])
    clear_prompt_btn_part.click(
        fn=lambda x: "", inputs=[], outputs=[dataset_description]
    )
    clear_btn_full.click(
        fn=lambda df: ("", "", pd.DataFrame(columns=df.columns)),
        inputs=[dataframe],
        outputs=[
            search_in,
            document_column,
        ],
    )

    app.load(fn=swap_visibility, outputs=main_ui)
    app.load(fn=get_org_dropdown, outputs=[org_name])
