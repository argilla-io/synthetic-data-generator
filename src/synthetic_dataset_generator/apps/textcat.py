import json
import random
import uuid
from typing import List, Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value
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
from synthetic_dataset_generator.constants import DEFAULT_BATCH_SIZE
from synthetic_dataset_generator.pipelines.base import get_rewritten_prompts
from synthetic_dataset_generator.pipelines.embeddings import (
    get_embeddings,
    get_sentence_embedding_dimensions,
)
from synthetic_dataset_generator.pipelines.textcat import (
    DEFAULT_DATASET_DESCRIPTIONS,
    generate_pipeline_code,
    get_labeller_generator,
    get_prompt_generator,
    get_textcat_generator,
)
from synthetic_dataset_generator.utils import (
    get_argilla_client,
    get_org_dropdown,
    get_preprocess_labels,
    get_random_repo_name,
    swap_visibility,
)


def _get_dataframe():
    return gr.Dataframe(
        headers=["labels", "text"],
        wrap=True,
        interactive=False,
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
    data = json.loads(result)
    system_prompt = data["classification_task"]
    labels = get_preprocess_labels(data["labels"])
    return system_prompt, labels


def generate_sample_dataset(
    system_prompt, difficulty, clarity, labels, multi_label, progress=gr.Progress()
):
    dataframe = generate_dataset(
        system_prompt=system_prompt,
        difficulty=difficulty,
        clarity=clarity,
        labels=labels,
        multi_label=multi_label,
        num_rows=10,
        progress=progress,
        is_sample=True,
    )
    return dataframe


def generate_dataset(
    system_prompt: str,
    difficulty: str,
    clarity: str,
    labels: List[str] = None,
    multi_label: bool = False,
    num_rows: int = 10,
    temperature: float = 0.9,
    is_sample: bool = False,
    progress=gr.Progress(),
) -> pd.DataFrame:
    num_rows = test_max_num_rows(num_rows)
    progress(0.0, desc="(1/2) Generating dataset")
    labels = get_preprocess_labels(labels)
    textcat_generator = get_textcat_generator(
        difficulty=difficulty,
        clarity=clarity,
        temperature=temperature,
        is_sample=is_sample,
    )
    updated_system_prompt = f"{system_prompt}. Optional labels: {', '.join(labels)}."
    if multi_label:
        updated_system_prompt = f"{updated_system_prompt}. Only apply relevant labels. Applying less labels is always better than applying too many labels."
    labeller_generator = get_labeller_generator(
        system_prompt=updated_system_prompt,
        labels=labels,
        multi_label=multi_label,
    )
    total_steps: int = num_rows * 2
    batch_size = DEFAULT_BATCH_SIZE

    # create text classification data
    n_processed = 0
    textcat_results = []
    rewritten_system_prompts = get_rewritten_prompts(system_prompt, num_rows)
    while n_processed < num_rows:
        progress(
            2 * 0.5 * n_processed / num_rows,
            total=total_steps,
            desc="(1/2) Generating dataset",
        )
        remaining_rows = num_rows - n_processed
        batch_size = min(batch_size, remaining_rows)
        inputs = []
        for _ in range(batch_size):
            k = 1
            if multi_label:
                num_labels = len(labels)
                k = int(
                    random.betavariate(alpha=(num_labels - 1), beta=num_labels)
                    * num_labels
                )
            sampled_labels = random.sample(labels, min(k, len(labels)))
            random.shuffle(sampled_labels)
            inputs.append(
                {
                    "task": f"{random.choice(rewritten_system_prompts)}. The text represents the following categories: {', '.join(sampled_labels)}"
                }
            )
        batch = list(textcat_generator.process(inputs=inputs))
        textcat_results.extend(batch[0])
        n_processed += batch_size
        random.seed(a=random.randint(0, 2**32 - 1))
    for result in textcat_results:
        result["text"] = result["input_text"]

    # label text classification data
    progress(2 * 0.5, desc="(2/2) Labeling dataset")
    n_processed = 0
    labeller_results = []
    while n_processed < num_rows:
        progress(
            0.5 + 0.5 * n_processed / num_rows,
            total=total_steps,
            desc="(2/2) Labeling dataset",
        )
        batch = textcat_results[n_processed : n_processed + batch_size]
        labels_batch = list(labeller_generator.process(inputs=batch))
        labeller_results.extend(labels_batch[0])
        n_processed += batch_size
        random.seed(a=random.randint(0, 2**32 - 1))
    progress(
        1,
        total=total_steps,
        desc="(2/2) Creating dataset",
    )

    # create final dataset
    distiset_results = []
    for result in labeller_results:
        record = {key: result[key] for key in ["labels", "text"] if key in result}
        distiset_results.append(record)

    dataframe = pd.DataFrame(distiset_results)
    if multi_label:

        def _validate_labels(x):
            if isinstance(x, str):  # single label
                return [x.lower().strip()]
            elif isinstance(x, list):  # multiple labels
                return list(
                    set(
                        label.lower().strip()
                        for label in x
                        if label.lower().strip() in labels
                    )
                )
            else:
                return list(set([random.choice(labels)]))

        dataframe["labels"] = dataframe["labels"].apply(_validate_labels)
        dataframe = dataframe[dataframe["labels"].notna()]
    else:

        def _validate_labels(x):
            if isinstance(x, str) and x.lower().strip() in labels:
                return x.lower().strip()
            elif isinstance(x, list):
                options = [
                    label.lower().strip()
                    for label in x
                    if isinstance(label, str) and label.lower().strip() in labels
                ]
                if options:
                    return random.choice(options)
                else:
                    return random.choice(labels)
            else:
                return random.choice(labels)

        dataframe = dataframe.rename(columns={"labels": "label"})
        dataframe["label"] = dataframe["label"].apply(_validate_labels)
    dataframe = dataframe[dataframe["text"].notna()]

    progress(1.0, desc="Dataset created")
    return dataframe


def push_dataset_to_hub(
    dataframe: pd.DataFrame,
    org_name: str,
    repo_name: str,
    multi_label: bool = False,
    labels: List[str] = None,
    oauth_token: Union[gr.OAuthToken, None] = None,
    private: bool = False,
    pipeline_code: str = "",
    progress=gr.Progress(),
):
    progress(0.0, desc="Validating")
    repo_id = validate_push_to_hub(org_name, repo_name)
    progress(0.3, desc="Preprocessing")
    labels = get_preprocess_labels(labels)
    progress(0.7, desc="Creating dataset")
    if multi_label:
        features = Features(
            {
                "text": Value("string"),
                "labels": Sequence(feature=ClassLabel(names=labels)),
            }
        )
    else:
        features = Features(
            {"text": Value("string"), "label": ClassLabel(names=labels)}
        )
    dataset = Dataset.from_pandas(
        dataframe.reset_index(drop=True),
        features=features,
    )
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


def push_dataset(
    org_name: str,
    repo_name: str,
    system_prompt: str,
    difficulty: str,
    clarity: str,
    multi_label: int = 1,
    num_rows: int = 10,
    labels: List[str] = None,
    private: bool = False,
    temperature: float = 0.8,
    pipeline_code: str = "",
    oauth_token: Union[gr.OAuthToken, None] = None,
    progress=gr.Progress(),
) -> pd.DataFrame:
    dataframe = generate_dataset(
        system_prompt=system_prompt,
        difficulty=difficulty,
        clarity=clarity,
        multi_label=multi_label,
        labels=labels,
        num_rows=num_rows,
        temperature=temperature,
    )
    push_dataset_to_hub(
        dataframe,
        org_name,
        repo_name,
        multi_label,
        labels,
        oauth_token,
        private,
        pipeline_code,
    )

    dataframe = dataframe[
        (dataframe["text"].str.strip() != "") & (dataframe["text"].notna())
    ]
    try:
        progress(0.1, desc="Setting up user and workspace")
        hf_user = HfApi().whoami(token=oauth_token.token)["name"]
        client = get_argilla_client()
        if client is None:
            return ""
        labels = get_preprocess_labels(labels)
        progress(0.5, desc="Creating dataset in Argilla")
        settings = rg.Settings(
            fields=[
                rg.TextField(
                    name="text",
                    description="The text classification data",
                    title="Text",
                ),
            ],
            questions=[
                (
                    rg.MultiLabelQuestion(
                        name="labels",
                        title="Labels",
                        description="The labels of the conversation",
                        labels=labels,
                    )
                    if multi_label
                    else rg.LabelQuestion(
                        name="label",
                        title="Label",
                        description="The label of the text",
                        labels=labels,
                    )
                ),
            ],
            metadata=[
                rg.IntegerMetadataProperty(name="text_length", title="Text Length"),
            ],
            vectors=[
                rg.VectorField(
                    name="text_embeddings",
                    dimensions=get_sentence_embedding_dimensions(),
                )
            ],
            guidelines="Please review the text and provide or correct the label where needed.",
        )

        dataframe["text_length"] = dataframe["text"].apply(len)
        dataframe["text_embeddings"] = get_embeddings(dataframe["text"].to_list())

        rg_dataset = client.datasets(name=repo_name, workspace=hf_user)
        if rg_dataset is None:
            rg_dataset = rg.Dataset(
                name=repo_name,
                workspace=hf_user,
                settings=settings,
                client=client,
            )
            rg_dataset = rg_dataset.create()
        progress(0.7, desc="Pushing dataset")
        hf_dataset = Dataset.from_pandas(dataframe)
        records = [
            rg.Record(
                fields={
                    "text": sample["text"],
                },
                metadata={"text_length": sample["text_length"]},
                vectors={"text_embeddings": sample["text_embeddings"]},
                suggestions=(
                    [
                        rg.Suggestion(
                            question_name="labels" if multi_label else "label",
                            value=(
                                sample["labels"] if multi_label else sample["label"]
                            ),
                        )
                    ]
                    if (
                        (not multi_label and sample["label"] in labels)
                        or (
                            multi_label
                            and all(label in labels for label in sample["labels"])
                        )
                    )
                    else None
                ),
            )
            for sample in hf_dataset
        ]
        rg_dataset.records.log(records=records)
        progress(1.0, desc="Dataset pushed")
    except Exception as e:
        raise gr.Error(f"Error pushing dataset to Argilla: {e}")
    return ""


def validate_input_labels(labels: List[str]) -> List[str]:
    if (
        not labels
        or len(set(label.lower().strip() for label in labels if label.strip())) < 2
    ):
        raise gr.Error(
            f"Please provide at least 2 unique, non-empty labels to classify your text. You provided {len(labels) if labels else 0}."
        )
    return labels


def show_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=True)}


def hide_pipeline_code_visibility():
    return {pipeline_code_ui: gr.Accordion(visible=False)}


######################
# Gradio UI
######################


with gr.Blocks() as app:
    with gr.Column() as main_ui:
        gr.Markdown("## 1. Describe the dataset you want")
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

        gr.HTML("<hr>")
        gr.Markdown("## 2. Configure your dataset")
        with gr.Row(equal_height=True):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    system_prompt = gr.Textbox(
                        label="System prompt",
                        placeholder="You are a helpful assistant.",
                        visible=True,
                    )
                    labels = gr.Dropdown(
                        choices=[],
                        allow_custom_value=True,
                        interactive=True,
                        label="Labels",
                        multiselect=True,
                        info="Add the labels to classify the text.",
                    )
                    multi_label = gr.Checkbox(
                        label="Multi-label",
                        value=False,
                        interactive=True,
                        info="If checked, the text will be classified into multiple labels.",
                    )
                    clarity = gr.Dropdown(
                        choices=[
                            ("Clear", "clear"),
                            (
                                "Understandable",
                                "understandable with some effort",
                            ),
                            ("Ambiguous", "ambiguous"),
                            ("Mixed", "mixed"),
                        ],
                        value="mixed",
                        label="Clarity",
                        info="Set how easily the correct label or labels can be identified.",
                        interactive=True,
                    )
                    difficulty = gr.Dropdown(
                        choices=[
                            ("High School", "high school"),
                            ("College", "college"),
                            ("PhD", "PhD"),
                            ("Mixed", "mixed"),
                        ],
                        value="high school",
                        label="Difficulty",
                        info="Select the comprehension level for the text. Ensure it matches the task context.",
                        interactive=True,
                    )
                    with gr.Row():
                        clear_btn_full = gr.Button("Clear", variant="secondary")
                        btn_apply_to_sample_dataset = gr.Button(
                            "Save", variant="primary"
                        )
                with gr.Column(scale=3):
                    dataframe = _get_dataframe()

        gr.HTML("<hr>")
        gr.Markdown("## 3. Generate your dataset")
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
                    value=0.8,
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
                        system_prompt.value,
                        difficulty=difficulty.value,
                        clarity=clarity.value,
                        labels=labels.value,
                        num_labels=len(labels.value) if multi_label.value else 1,
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
        outputs=[system_prompt, labels],
        show_progress=True,
    ).then(
        fn=generate_sample_dataset,
        inputs=[system_prompt, difficulty, clarity, labels, multi_label],
        outputs=[dataframe],
        show_progress=True,
    )

    btn_apply_to_sample_dataset.click(
        fn=validate_input_labels,
        inputs=[labels],
        outputs=[labels],
        show_progress=True,
    ).success(
        fn=generate_sample_dataset,
        inputs=[system_prompt, difficulty, clarity, labels, multi_label],
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
        fn=validate_input_labels,
        inputs=[labels],
        outputs=[labels],
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
            system_prompt,
            difficulty,
            clarity,
            multi_label,
            num_rows,
            labels,
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
        inputs=[
            system_prompt,
            difficulty,
            clarity,
            labels,
            multi_label,
            num_rows,
        ],
        outputs=[pipeline_code],
    ).success(
        fn=show_pipeline_code_visibility,
        inputs=[],
        outputs=[pipeline_code_ui],
    )

    gr.on(
        triggers=[clear_btn_part.click, clear_btn_full.click],
        fn=lambda _: (
            "",
            "",
            [],
            _get_dataframe(),
        ),
        inputs=[dataframe],
        outputs=[dataset_description, system_prompt, labels, multi_label, dataframe],
    )

    app.load(fn=swap_visibility, outputs=main_ui)
    app.load(fn=get_org_dropdown, outputs=[org_name])
    app.load(fn=get_random_repo_name, outputs=[repo_name])
