import io
import uuid
from tqdm import tqdm
from typing import Union

import argilla as rg
import gradio as gr
import pandas as pd
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, get_dataset_split_names, load_dataset
from gradio import OAuthToken
from huggingface_hub import HfApi, upload_file, repo_exists
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from synthetic_dataset_generator.constants import MAX_NUM_ROWS
from synthetic_dataset_generator.utils import get_argilla_client


def validate_argilla_user_workspace_dataset(
    dataset_name: str,
    add_to_existing_dataset: bool = True,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
) -> str:
    progress(0.1, desc="Validating dataset configuration")
    hf_user = HfApi().whoami(token=oauth_token.token)["name"]
    client = get_argilla_client()
    if dataset_name is None or dataset_name == "":
        raise gr.Error("Dataset name is required")
    # Create user if it doesn't exist
    rg_user = client.users(username=hf_user)
    if rg_user is None:
        rg_user = client.users.add(
            rg.User(username=hf_user, role="admin", password=str(uuid.uuid4()))
        )
    # Create workspace if it doesn't exist
    workspace = client.workspaces(name=hf_user)
    if workspace is None:
        workspace = client.workspaces.add(rg.Workspace(name=hf_user))
        workspace.add_user(hf_user)
    # Check if dataset exists
    dataset = client.datasets(name=dataset_name, workspace=hf_user)
    if dataset and not add_to_existing_dataset:
        raise gr.Error(f"Dataset {dataset_name} already exists")
    progress(1.0, desc="Dataset configuration validated")
    return ""


def push_pipeline_code_to_hub(
    pipeline_code: str,
    org_name: str,
    repo_name: str,
    oauth_token: Union[OAuthToken, None] = None,
    progress=gr.Progress(),
):
    repo_id: str | None = validate_push_to_hub(org_name, repo_name)
    progress(0.1, desc="Uploading pipeline code")
    with io.BytesIO(pipeline_code.encode("utf-8")) as f:
        upload_file(
            path_or_fileobj=f,
            path_in_repo="pipeline.py",
            repo_id=repo_id,
            repo_type="dataset",
            token=oauth_token.token,
            commit_message="Include pipeline script",
            create_pr=False,
        )
    progress(1.0, desc="Pipeline code uploaded")


def validate_push_to_hub(org_name: str, repo_name: str):
    repo_id = (
        f"{org_name}/{repo_name}"
        if repo_name is not None and org_name is not None
        else None
    )
    if repo_id is not None:
        if not all([repo_id, org_name, repo_name]):
            raise gr.Error(
                "Please provide a `repo_name` and `org_name` to push the dataset to."
            )
    return repo_id


def combine_datasets(
    repo_id: str, dataset: Dataset, oauth_token: Union[OAuthToken, None]
) -> Dataset:
    try:
        new_dataset = load_dataset(
            repo_id,
            split="train",
            download_mode="force_redownload",
            token=oauth_token.token,
        )
        return concatenate_datasets([dataset, new_dataset])
    except Exception:
        return dataset


def show_success_message(org_name: str, repo_name: str) -> gr.Markdown:
    client = get_argilla_client()
    if client is None:
        return gr.Markdown(
            value=f"""
                <div style="padding: 1em; background-color: var(--block-background-fill); border-color: var(--border-color-primary); border-width: 1px; border-radius: 5px;">
                    <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                    <p style="margin-top: 0.5em;">
                        The generated dataset is in the right format for fine-tuning with TRL, AutoTrain, or other frameworks.
                        <div style="display: flex; gap: 10px;">
                            <a href="https://huggingface.co/datasets/{org_name}/{repo_name}" target="_blank" class="lg primary svelte-1137axg" style="color: white !important; margin-top: 0.5em; text-decoration: none;">
                                Open in Hugging Face
                            </a>
                        </div>
                    </p>
                    <p style="margin-top: 1em; color: var(--block-title-text-color)">
                        By configuring an `ARGILLA_API_URL` and `ARGILLA_API_KEY` you can curate the dataset in Argilla.
                        Unfamiliar with Argilla? Here are some docs to help you get started:
                        <br>• <a href="https://docs.argilla.io/latest/getting_started/quickstart/" target="_blank">How to get started with Argilla</a>
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/annotate/" target="_blank">How to curate data in Argilla</a>
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/import_export/" target="_blank">How to export data once you have reviewed the dataset</a>
                    </p>
                </div>
                """,
            visible=True,
            height=None,
            min_height=None,
            max_height=None,
        )
    argilla_api_url = client.api_url
    return gr.Markdown(
        value=f"""
                <div style="padding: 1em; background-color: var(--block-background-fill); border-color: var(--border-color-primary); border-width: 1px; border-radius: 5px;">
                    <h3 style="color: #2e7d32; margin: 0;">Dataset Published Successfully!</h3>
                    <p style="margin-top: 0.5em;">
                        The generated dataset is <a href="https://huggingface.co/datasets/{org_name}/{repo_name}" target="_blank">available in the Hub</a>. It is in the right format for fine-tuning with TRL, AutoTrain, or other frameworks.
                        <div style="display: flex; gap: 10px;">
                            <a href="{argilla_api_url}" target="_blank" class="lg primary svelte-1137axg" style="color: white !important; margin-top: 0.5em; text-decoration: none;">
                                Open in Argilla
                            </a>
                        </div>
                    </p>
                    <p style="margin-top: 1em; color: var(--block-title-text-color)">
                        Unfamiliar with Argilla? Here are some docs to help you get started:
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/annotate/" target="_blank">How to curate data in Argilla</a>
                        <br>• <a href="https://docs.argilla.io/latest/how_to_guides/import_export/" target="_blank">How to export data once you have reviewed the dataset</a>
                    </p>
                </div>
            """,
        visible=True,
        height=None,
        min_height=None,
        max_height=None,
    )


def hide_success_message() -> gr.Markdown:
    return gr.Markdown(value="", visible=True, height=100)


def test_max_num_rows(num_rows: int) -> int:
    if num_rows > MAX_NUM_ROWS:
        num_rows = MAX_NUM_ROWS
        gr.Info(
            f"Number of rows is larger than the configured maximum. Setting number of rows to {MAX_NUM_ROWS}. Set environment variable `MAX_NUM_ROWS` to change this behavior."
        )
    return num_rows


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


def load_dataset_from_hub(
    repo_id: str,
    num_rows: int = 10,
    token: Union[OAuthToken, None] = None,
    progress=gr.Progress(track_tqdm=True),
):
    if not repo_id:
        raise gr.Error("Please provide a Hub repo ID")
    subsets = get_dataset_config_names(repo_id, token=token)
    splits = get_dataset_split_names(repo_id, subsets[0], token=token)
    ds = load_dataset(repo_id, subsets[0], split=splits[0], token=token, streaming=True)
    rows = []
    for idx, row in enumerate(tqdm(ds, desc="Loading the dataset", total=num_rows)):
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
            multiselect=False,
        ),
    )


def preprocess_input_data(
    file_paths: list[str], num_rows: int, progress=gr.Progress(track_tqdm=True)
):
    if not file_paths:
        raise gr.Error("Please provide an input file")

    data = {}
    total_chunks = 0

    for file_path in tqdm(file_paths, desc="Processing files", total=len(file_paths)):
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
            choices=["chunks"],
            label="Documents column",
            value=col_doc,
            interactive=(False if col_doc == "" else True),
            multiselect=False,
        ),
    )
