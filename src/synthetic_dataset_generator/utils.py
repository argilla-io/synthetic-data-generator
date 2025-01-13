import json
import uuid
import warnings
from typing import List, Optional, Union

import argilla as rg
import gradio as gr
import numpy as np
import pandas as pd
from gradio.oauth import (
    OAuthToken,
    get_space,
)
from huggingface_hub import whoami
from jinja2 import Environment, meta

from synthetic_dataset_generator.constants import argilla_client


def get_duplicate_button():
    if get_space() is not None:
        return gr.DuplicateButton(size="lg")


def list_orgs(oauth_token: Union[OAuthToken, None] = None):
    if oauth_token is None:
        return []
    try:
        data = whoami(oauth_token.token)
    except Exception:
        swap_visibility(None)
        return []

    try:
        if data["auth"]["type"] == "oauth":
            organizations = [data["name"]] + [org["name"] for org in data["orgs"]]
        elif data["auth"]["type"] == "access_token":
            organizations = [data["name"]] + [org["name"] for org in data["orgs"]]
        else:
            organizations = [
                entry["entity"]["name"]
                for entry in data["auth"]["accessToken"]["fineGrained"]["scoped"]
                if "repo.write" in entry["permissions"]
            ]
            organizations = [org for org in organizations if org != data["name"]]
            organizations = [data["name"]] + organizations
    except Exception as e:
        warnings.warn(str(e))
        gr.Info(
            "Your user token does not have the necessary permissions to push to organizations."
            "Please check your OAuth permissions in https://huggingface.co/settings/connected-applications."
            "Update your token permissions to include repo.write: https://huggingface.co/settings/tokens."
        )
        return []

    return organizations


def get_random_repo_name():
    return f"my-distiset-{str(uuid.uuid4())[:8]}"


def get_org_dropdown(oauth_token: Union[OAuthToken, None] = None):
    if oauth_token is not None:
        orgs = list_orgs(oauth_token)
    else:
        orgs = []
    return gr.Dropdown(
        label="Organization",
        choices=orgs,
        value=orgs[0] if orgs else None,
        allow_custom_value=True,
        interactive=True,
    )


def swap_visibility(oauth_token: Union[OAuthToken, None]):
    if oauth_token:
        return gr.update(elem_classes=["main_ui_logged_in"])
    else:
        return gr.update(elem_classes=["main_ui_logged_out"])


def get_argilla_client() -> Union[rg.Argilla, None]:
    return argilla_client


def get_preprocess_labels(labels: Optional[List[str]]) -> List[str]:
    return list(set([label.lower().strip() for label in labels])) if labels else []


def column_to_list(dataframe: pd.DataFrame, column_name: str) -> List[str]:
    if column_name in dataframe.columns:
        return dataframe[column_name].tolist()
    else:
        raise ValueError(f"Column '{column_name}' does not exist.")


def process_columns(
    dataframe,
    instruction_column: str,
    response_columns: Union[str, List[str]],
) -> List[dict]:
    instruction_column = [instruction_column]
    if isinstance(response_columns, str):
        response_columns = [response_columns]

    data = []
    for _, row in dataframe.iterrows():
        instruction = ""
        for col in instruction_column:
            value = row[col]
            if isinstance(value, (list, np.ndarray)):
                user_contents = [d["content"] for d in value if d.get("role") == "user"]
                if user_contents:
                    instruction = user_contents[-1]
            elif isinstance(value, str):
                try:
                    parsed_message = json.loads(value)
                    user_contents = [
                        d["content"] for d in parsed_message if d.get("role") == "user"
                    ]
                    if user_contents:
                        instruction = user_contents[-1]
                except json.JSONDecodeError:
                    instruction = value
            else:
                instruction = ""

        generations = []
        for col in response_columns:
            value = row[col]
            if isinstance(value, (list, np.ndarray)):
                if all(isinstance(item, dict) and "role" in item for item in value):
                    assistant_contents = [
                        d["content"] for d in value if d.get("role") == "assistant"
                    ]
                    if assistant_contents:
                        generations.append(assistant_contents[-1])
                else:
                    generations.extend(value)
            elif isinstance(value, str):
                try:
                    parsed_message = json.loads(value)
                    assistant_contents = [
                        d["content"]
                        for d in parsed_message
                        if d.get("role") == "assistant"
                    ]
                    if assistant_contents:
                        generations.append(assistant_contents[-1])
                except json.JSONDecodeError:
                    generations.append(value)
            else:
                pass

        data.append({"instruction": instruction, "generations": generations})

    return data


def extract_column_names(prompt_template: str) -> List[str]:
    env = Environment()
    parsed_content = env.parse(prompt_template)
    variables = meta.find_undeclared_variables(parsed_content)
    return list(variables)


def pad_or_truncate_list(lst, target_length):
    lst = lst or []
    lst_length = len(lst)
    if lst_length >= target_length:
        return lst[-target_length:]
    else:
        return lst + [None] * (target_length - lst_length)
