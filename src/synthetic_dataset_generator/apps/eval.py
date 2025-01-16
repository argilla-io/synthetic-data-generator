import json
import uuid
from typing import Union

import argilla as rg
import gradio as gr
import numpy as np
import pandas as pd
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from distilabel.distiset import Distiset
from gradio.oauth import OAuthToken  #
from gradio_huggingfacehub_search import HuggingfaceHubSearch
from huggingface_hub import HfApi, repo_exists

from synthetic_dataset_generator.apps.base import (
    combine_datasets,
    get_iframe,
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
from synthetic_dataset_generator.pipelines.eval import (
    generate_pipeline_code,
    get_custom_evaluator,
    get_ultrafeedback_evaluator,
)
from synthetic_dataset_generator.utils import (
    column_to_list,
    extract_column_names,
    get_argilla_client,
    get_org_dropdown,
    get_random_repo_name,
    pad_or_truncate_list,
    process_columns,
    swap_visibility,
)


def get_valid_columns(dataframe: pd.DataFrame):
    instruction_valid_columns = []
    response_valid_columns = []

    for col in dataframe.columns:
        sample_val = dataframe[col].iloc[0]
        if isinstance(sample_val, str) or (
            isinstance(sample_val, (list, np.ndarray))
            and all(isinstance(item, dict) and "role" in item for item in sample_val)
        ):
            instruction_valid_columns.append(col)
            response_valid_columns.append(col)
        if isinstance(sample_val, (list, np.ndarray)) and all(
            isinstance(item, str) for item in sample_val
        ):
            response_valid_columns.append(col)

    return instruction_valid_columns, response_valid_columns


def load_dataset_from_hub(
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
    instruction_valid_columns, response_valid_columns = get_valid_columns(dataframe)
    col_instruction = instruction_valid_columns[0] if instruction_valid_columns else ""
    col_response = "No valid response columns found."
    for col in response_valid_columns:
        if col != col_instruction:
            col_response = col
            break

    prompt_template = gr.Code(
        label="Prompt template",
        value="\n".join(
            [
                "Evaluate the following text based on criteria.",
                "Criteria: quality.",
                "Score: between 1 and 10.",
                "Text: {{" + col_response + "}}",
            ]
        ),
        language="jinja2",
        interactive=True,
    )
    structured_output = gr.Code(
        label="Structured output",
        value=json.dumps(
            {
                "type": "object",
                "properties": {"quality": {"type": "integer"}},
                "required": ["quality"],
            },
            indent=4,
        ),
        language="json",
        interactive=True,
    )
    return (
        dataframe,
        gr.Dropdown(
            choices=instruction_valid_columns,
            label="Instruction column",
            value=col_instruction,
            interactive=True,
        ),
        gr.Dropdown(
            choices=response_valid_columns,
            label="Response column",
            value=col_response,
            interactive=False
            if col_response == "No valid response columns found."
            else True,
        ),
        prompt_template,
        structured_output,
    )


def define_evaluation_aspects(task_type: str):
    if task_type == "chat-eval":
        return gr.Dropdown(
            value=["overall-rating"],
            choices=["helpfulness", "truthfulness", "overall-rating", "honesty"],
            label="Evaluation Aspects",
            multiselect=True,
            interactive=True,
        )
    else:
        return gr.Dropdown(interactive=False, visible=False)


def evaluate_instruction_response(
    dataframe: pd.DataFrame,
    aspects: list[str],
    instruction_column: str,
    response_columns: str,
    num_rows: int = 10,
    is_sample: bool = False,
    progress=gr.Progress(),
):
    progress(0.0, desc="Evaluating instructions and responses")
    data = process_columns(dataframe, instruction_column, response_columns)
    num_generations = len(data[0]["generations"])
    evaluated_results = []
    for entry in data:
        result_row = {
            "instruction": entry["instruction"],
            "generations": entry["generations"],
        }
        for aspect in aspects:
            result_row[f"ratings_{aspect}"] = None
            result_row[f"rationale_for_ratings_{aspect}"] = None
            if aspect in ["truthfulness", "helpfulness"]:
                result_row[f"type_{aspect}"] = None
                result_row[f"rationale_for_type_{aspect}"] = None
        result_row["model_name"] = None
        evaluated_results.append(result_row)

    batch_size = DEFAULT_BATCH_SIZE
    total_steps: int = len(aspects) * num_rows

    # evaluate instructions and responses
    for aspect in aspects:
        ultrafeedback_evaluator = get_ultrafeedback_evaluator(aspect, is_sample)
        n_processed = 0

        while n_processed < num_rows:
            progress(
                (len(aspects) * n_processed) / total_steps,
                total=total_steps,
                desc=f"Evaluating aspect: {aspect}",
            )

            remaining_rows = num_rows - n_processed
            batch_size = min(batch_size, remaining_rows)
            inputs = data[n_processed : n_processed + batch_size]
            batch_results = list(ultrafeedback_evaluator.process(inputs=inputs))
            for j, result in enumerate(batch_results[0]):
                idx = n_processed + j
                evaluated_results[idx][f"ratings_{aspect}"] = pad_or_truncate_list(
                    result.get("ratings"), num_generations
                )
                evaluated_results[idx]["model_name"] = result.get("model_name")
                if aspect in ["truthfulness", "helpfulness"]:
                    evaluated_results[idx][f"type_{aspect}"] = pad_or_truncate_list(
                        result.get("types"), num_generations
                    )
                    evaluated_results[idx][f"rationale_for_type_{aspect}"] = (
                        pad_or_truncate_list(result.get("rationales"), num_generations)
                    )
                    evaluated_results[idx][f"rationale_for_ratings_{aspect}"] = (
                        pad_or_truncate_list(
                            result.get("rationales-for-ratings"), num_generations
                        )
                    )
                else:
                    evaluated_results[idx][f"rationale_for_ratings_{aspect}"] = (
                        pad_or_truncate_list(result.get("rationales"), num_generations)
                    )
            n_processed += batch_size

    # create final dataset
    dataframe = pd.DataFrame(evaluated_results)
    progress(1.0, desc="Dataset evaluation completed")
    return dataframe


def evaluate_custom(
    dataframe: pd.DataFrame,
    prompt_template: str,
    structured_output: dict,
    num_rows: int = 10,
    is_sample: bool = False,
    progress=gr.Progress(),
):
    progress(0.0, desc="Evaluating dataset")
    columns = extract_column_names(prompt_template)
    input_columns = {column: column_to_list(dataframe, column) for column in columns}

    custom_evaluator = get_custom_evaluator(
        prompt_template, structured_output, columns, is_sample
    )
    batch_size = DEFAULT_BATCH_SIZE

    # evaluate the data
    n_processed = 0
    evaluation_results = []
    while n_processed < num_rows:
        progress(
            n_processed / num_rows,
            desc="Evaluating dataset",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)

        inputs = []
        for idx in range(n_processed, n_processed + batch_size):
            input = {column: input_columns[column][idx] for column in input_columns}
            inputs.append(input)

        batch = list(custom_evaluator.process(inputs=inputs))
        evaluation_results.extend(batch[0])
        n_processed += batch_size

    # create final dataset
    distiset_results = []
    for result in evaluation_results:
        record = {key: result[key] for key in result if key != "distilabel_metadata"}
        distiset_results.append(record)

    dataframe = pd.DataFrame(distiset_results)
    progress(1.0, desc="Dataset evaluation completed")
    return dataframe


def _evaluate_dataset(
    dataframe: pd.DataFrame,
    eval_type: str,
    aspects_instruction_response: list[str],
    instruction_instruction_response: str,
    response_instruction_response: str,
    prompt_template: str,
    structured_output: dict,
    num_rows: int = 10,
    is_sample: bool = False,
):
    num_rows = test_max_num_rows(num_rows)
    if eval_type == "chat-eval":
        dataframe = evaluate_instruction_response(
            dataframe=dataframe,
            aspects=aspects_instruction_response,
            instruction_column=instruction_instruction_response,
            response_columns=response_instruction_response,
            num_rows=num_rows,
            is_sample=is_sample,
        )
    else:
        dataframe = evaluate_custom(
            dataframe=dataframe,
            prompt_template=prompt_template,
            structured_output=structured_output,
            num_rows=num_rows,
            is_sample=is_sample,
        )
    return dataframe


def evaluate_sample_dataset(
    repo_id: str,
    eval_type: str,
    aspects_instruction_response: list[str],
    instruction_instruction_response: str,
    response_instruction_response: str,
    prompt_template: str,
    structured_output: dict,
):
    dataframe, _, _, _, _ = load_dataset_from_hub(repo_id, num_rows=10)
    dataframe = _evaluate_dataset(
        dataframe=dataframe,
        eval_type=eval_type,
        aspects_instruction_response=aspects_instruction_response,
        instruction_instruction_response=instruction_instruction_response,
        response_instruction_response=response_instruction_response,
        prompt_template=prompt_template,
        structured_output=structured_output,
        num_rows=10,
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
    num_rows: int,
    original_repo_id: str,
    eval_type: str,
    aspects_instruction_response: list[str],
    instruction_instruction_response: str,
    response_instruction_response: str,
    prompt_template: str,
    structured_output: dict,
    pipeline_code: str,
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    dataframe, _, _, _, _ = load_dataset_from_hub(original_repo_id, num_rows=num_rows)
    dataframe = _evaluate_dataset(
        dataframe=dataframe,
        eval_type=eval_type,
        aspects_instruction_response=aspects_instruction_response,
        instruction_instruction_response=instruction_instruction_response,
        response_instruction_response=response_instruction_response,
        prompt_template=prompt_template,
        structured_output=structured_output,
        num_rows=num_rows,
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
        if eval_type == "chat-eval":
            num_generations = len((dataframe["generations"][0]))
            fields = [
                rg.ChatField(
                    name=f"chat_{i}",
                    title=f"Chat {i+1}",
                    description=f"User and assistant conversation for generation {i+1}",
                )
                for i in range(num_generations)
            ]
            questions = []
            for i in range(num_generations):
                for aspect in aspects_instruction_response:
                    questions.append(
                        rg.RatingQuestion(
                            name=f"ratings_{aspect}_{i}",
                            values=list(range(11)),
                            title=f"Ratings for {aspect} for response {i+1}",
                            required=True,
                        )
                    )
                    questions.append(
                        rg.TextQuestion(
                            name=f"rationale_for_ratings_{aspect}_{i}",
                            title=f"Rationale for ratings for {aspect} for response {i+1}",
                            required=False,
                            use_markdown=True,
                        )
                    )
                    if aspect in ["truthfulness", "helpfulness"]:
                        questions.append(
                            rg.RatingQuestion(
                                name=f"type_{aspect}_{i}",
                                values=list(range(1, 6)),
                                title=f"The type of the response {i+1} for {aspect}",
                                required=True,
                            )
                        )
                        questions.append(
                            rg.TextQuestion(
                                name=f"rationale_for_type_{aspect}_{i}",
                                title=f"Rationale for type of the response {i+1} for {aspect}",
                                required=False,
                                use_markdown=True,
                            )
                        )
            metadata = [
                rg.IntegerMetadataProperty(
                    name="instruction_length", title="Instruction length"
                ),
            ]
            for i in range(num_generations):
                metadata.append(
                    rg.IntegerMetadataProperty(
                        name=f"response_{i}_length", title=f"Response {i+1} length"
                    )
                )
            vectors = [
                rg.VectorField(
                    name="instruction_embeddings",
                    dimensions=get_sentence_embedding_dimensions(),
                )
            ]
            settings = rg.Settings(
                fields=fields,
                questions=questions,
                metadata=metadata,
                vectors=vectors,
                guidelines="Please review the conversation and provide an evaluation.",
            )

            dataframe["instruction_length"] = dataframe["instruction"].apply(len)
            for i in range(num_generations):
                dataframe[f"response_{i}_length"] = dataframe["generations"].apply(
                    lambda gens: len(gens[i]) if i < len(gens) else 0
                )
            dataframe["instruction_embeddings"] = get_embeddings(
                dataframe["instruction"].to_list()
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
            records = []
            for sample in hf_dataset:
                fields = {}
                metadata = {"instruction_length": sample.get("instruction_length", 0)}
                vectors = {
                    "instruction_embeddings": sample.get("instruction_embeddings", [])
                }
                suggestions = []
                generations = sample.get("generations", [])
                for i in range(num_generations):
                    fields[f"chat_{i}"] = [
                        {"role": "user", "content": sample.get("instruction", "")},
                        {"role": "assistant", "content": generations[i]},
                    ]
                    metadata[f"response_{i}_length"] = sample.get(
                        f"response_{i}_length", 0
                    )

                    for aspect in aspects_instruction_response:
                        ratings = sample.get(f"ratings_{aspect}", [])
                        rationales = sample.get(f"rationale_for_ratings__{aspect}", [])

                        rating_value = (
                            ratings[i]
                            if ratings and isinstance(ratings[i], int)
                            else None
                        )
                        rationale_value = (
                            rationales[i]
                            if rationales and isinstance(rationales[i], str)
                            else None
                        )

                        if rating_value is not None:
                            suggestions.append(
                                rg.Suggestion(
                                    question_name=f"ratings_{aspect}_{i}",
                                    value=rating_value,
                                )
                            )
                        if rationale_value is not None:
                            suggestions.append(
                                rg.Suggestion(
                                    question_name=f"rationale_for_ratings_{aspect}_{i}",
                                    value=rationale_value,
                                )
                            )

                        if aspect in ["truthfulness", "helpfulness"]:
                            types = sample.get(f"type_{aspect}", [])
                            rationale_types = sample.get(
                                f"rationale_for_type_{aspect}", []
                            )

                            type_value = (
                                types[i]
                                if types and isinstance(types[i], int)
                                else None
                            )
                            rationale_type_value = (
                                rationale_types[i]
                                if rationale_types
                                and isinstance(rationale_types[i], str)
                                else None
                            )
                            if type_value is not None:
                                suggestions.append(
                                    rg.Suggestion(
                                        question_name=f"type_{aspect}_{i}",
                                        value=type_value,
                                    )
                                )
                            if rationale_type_value is not None:
                                suggestions.append(
                                    rg.Suggestion(
                                        question_name=f"rationale_for_type_{aspect}_{i}",
                                        value=rationale_type_value,
                                    )
                                )
                records.append(
                    rg.Record(
                        fields=fields,
                        metadata=metadata,
                        vectors=vectors,
                        suggestions=suggestions,
                    )
                )
            rg_dataset.records.log(records=records)
            progress(1.0, desc="Dataset pushed to Argilla")
        else:
            columns = extract_column_names(prompt_template)
            settings = rg.Settings(
                fields=[
                    rg.TextField(
                        name=column,
                        title=column.capitalize(),
                        description="The column content",
                    )
                    for column in columns
                ],
                questions=[
                    rg.TextQuestion(
                        name="evaluation",
                        title="Evaluation",
                        description="The generated evaluation",
                        use_markdown=True,
                    ),
                ],
                metadata=[
                    rg.IntegerMetadataProperty(
                        name=f"{column}_length", title=f"{column.capitalize()} length"
                    )
                    for column in columns
                ],
                vectors=[
                    rg.VectorField(
                        name=f"{column}_embeddings",
                        dimensions=get_sentence_embedding_dimensions(),
                    )
                    for column in columns
                ],
                guidelines="Please review, correct and provide an accurate evaluation.",
            )
            for column in columns:
                dataframe[f"{column}_length"] = dataframe[column].apply(len)
                dataframe[f"{column}_embeddings"] = get_embeddings(dataframe[column])

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
            rg_dataset.records.log(
                records=hf_dataset, mapping={"generation": "evaluation"}
            )
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
        gr.Markdown("## 1. Select your input dataset")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                search_in = HuggingfaceHubSearch(
                    label="Search",
                    placeholder="Search for a dataset",
                    search_type="dataset",
                    sumbit_on_select=True,
                )
                with gr.Row():
                    clear_btn_part = gr.Button("Clear", variant="secondary")
                    load_btn = gr.Button("Load", variant="primary")

            with gr.Column(scale=3):
                examples = gr.Examples(
                    examples=[
                        "argilla/distilabel-sft-easy",
                        "HuggingFaceFW/fineweb-edu",
                        "argilla/distilabel-intel-orca-dpo-pairs",
                    ],
                    label="Example datasets",
                    fn=lambda x: x,
                    inputs=[search_in],
                    run_on_click=True,
                )
                search_out = gr.HTML(label="Dataset preview", visible=False)

        gr.HTML(value="<hr>")
        gr.Markdown(value="## 2. Configure your task")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                eval_type = gr.Dropdown(
                    label="Evaluation type",
                    choices=["chat-eval", "custom-eval"],
                    value="chat-eval",
                    multiselect=False,
                    visible=False,
                )
                with gr.Tab("Response Evaluation") as tab_instruction_response:
                    aspects_instruction_response = define_evaluation_aspects(
                        "chat-eval"
                    )
                    instruction_instruction_response = gr.Dropdown(
                        label="Instruction Column",
                        info="Select the instruction column to evaluate",
                        choices=["Load your data first in step 1."],
                        value="Load your data first in step 1.",
                        interactive=False,
                        multiselect=False,
                        allow_custom_value=False,
                    )
                    response_instruction_response = gr.Dropdown(
                        label="Response Column",
                        info="Select the response column(s) to evaluate",
                        choices=["Load your data first in step 1."],
                        value="Load your data first in step 1.",
                        interactive=False,
                        multiselect=False,
                        allow_custom_value=False,
                    )
                    tab_instruction_response.select(
                        fn=lambda: "chat-eval",
                        inputs=[],
                        outputs=[eval_type],
                    )
                with gr.Tab("Custom Evaluation Prompt") as tab_custom:
                    aspects_custom = define_evaluation_aspects("custom-eval")
                    prompt_template = gr.Code(
                        label="Prompt template",
                        value="Load your data first in step 1.",
                        language="markdown",
                        interactive=False,
                    )
                    structured_output = gr.Code(
                        label="Structured output",
                        value="Load your data first in step 1.",
                        language="json",
                        interactive=False,
                    )
                    tab_custom.select(
                        fn=lambda: "custom-eval",
                        inputs=[],
                        outputs=[eval_type],
                    )
                with gr.Row():
                    clear_btn_full = gr.Button("Clear", variant="secondary")
                    btn_apply_to_sample_dataset = gr.Button("Save", variant="primary")
            with gr.Column(scale=3):
                dataframe = gr.Dataframe(
                    headers=["prompt", "completion", "evaluation"],
                    wrap=True,
                    interactive=False,
                )

        gr.HTML(value="<hr>")
        gr.Markdown(value="## 3. Evaluate your dataset")
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
                        aspects=aspects_instruction_response.value,
                        instruction_column=instruction_instruction_response,
                        response_columns=response_instruction_response,
                        prompt_template=prompt_template.value,
                        structured_output=structured_output.value,
                        num_rows=num_rows.value,
                        eval_type=eval_type.value,
                    )
                    pipeline_code = gr.Code(
                        value=code,
                        language="python",
                        label="Distilabel Pipeline Code",
                    )

    search_in.submit(fn=get_iframe, inputs=search_in, outputs=search_out).then(
        fn=lambda df: pd.DataFrame(columns=df.columns),
        inputs=[dataframe],
        outputs=[dataframe],
    )

    load_btn.click(
        fn=load_dataset_from_hub,
        inputs=[search_in],
        outputs=[
            dataframe,
            instruction_instruction_response,
            response_instruction_response,
            prompt_template,
            structured_output,
        ],
    )

    btn_apply_to_sample_dataset.click(
        fn=evaluate_sample_dataset,
        inputs=[
            search_in,
            eval_type,
            aspects_instruction_response,
            instruction_instruction_response,
            response_instruction_response,
            prompt_template,
            structured_output,
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
            num_rows,
            search_in,
            eval_type,
            aspects_instruction_response,
            instruction_instruction_response,
            response_instruction_response,
            prompt_template,
            structured_output,
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
            prompt_template,
            structured_output,
            eval_type,
        ],
        outputs=[pipeline_code],
    ).success(
        fn=show_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    )

    clear_btn_part.click(fn=lambda : "", inputs=[], outputs=[search_in])
    clear_btn_full.click(
        fn=lambda df: ("", "", pd.DataFrame(columns=df.columns)),
        inputs=[dataframe],
        outputs=[
            instruction_instruction_response,
            response_instruction_response,
        ],
    )

    app.load(fn=swap_visibility, outputs=main_ui)
    app.load(fn=get_org_dropdown, outputs=[org_name])
    app.load(fn=get_random_repo_name, outputs=[repo_name])
