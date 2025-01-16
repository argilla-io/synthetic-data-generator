from typing import Optional

import distilabel
import distilabel.distiset
import gradio as gr
from distilabel.utils.card.dataset_card import (
    DistilabelDatasetCard,
    size_categories_parser,
)
from huggingface_hub import DatasetCardData, HfApi


class CustomDistisetWithAdditionalTag(distilabel.distiset.Distiset):
    def _generate_card(
        self,
        repo_id: str,
        token: str,
        include_script: bool = False,
        filename_py: Optional[str] = None,
    ) -> None:
        """Generates a dataset card and pushes it to the Hugging Face Hub, and
        if the `pipeline.yaml` path is available in the `Distiset`, uploads that
        to the same repository.

        Args:
            repo_id: The ID of the repository to push to, from the `push_to_hub` method.
            token: The token to authenticate with the Hugging Face Hub, from the `push_to_hub` method.
            include_script: Whether to upload the script to the hugging face repository.
            filename_py: The name of the script. If `include_script` is True, the script will
                be uploaded to the repository using this name, otherwise it won't be used.
        """
        card = self._get_card(
            repo_id=repo_id,
            token=token,
            include_script=include_script,
            filename_py=filename_py,
        )

        card.push_to_hub(
            repo_id,
            repo_type="dataset",
            token=token,
        )
        if self.pipeline_path:
            # If the pipeline.yaml is available, upload it to the Hugging Face Hub as well.
            HfApi().upload_file(
                path_or_fileobj=self.pipeline_path,
                path_in_repo=distilabel.distiset.PIPELINE_CONFIG_FILENAME,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

    def _get_card(
        self,
        repo_id: str,
        token: Optional[str] = None,
        include_script: bool = False,
        filename_py: Optional[str] = None,
    ) -> DistilabelDatasetCard:
        """Generates the dataset card for the `Distiset`.

        Note:
            If `repo_id` and `token` are provided, it will extract the metadata from the README.md file
            on the hub.

        Args:
            repo_id: Name of the repository to push to, or the path for the distiset if saved to disk.
            token: The token to authenticate with the Hugging Face Hub.
                We assume that if it's provided, the dataset will be in the Hugging Face Hub,
                so the README metadata will be extracted from there.
            include_script: Whether to upload the script to the hugging face repository.
            filename_py: The name of the script. If `include_script` is True, the script will
                be uploaded to the repository using this name, otherwise it won't be used.

        Returns:
            The dataset card for the `Distiset`.
        """
        sample_records = {}
        for name, dataset in self.items():
            sample_records[name] = (
                dataset[0] if not isinstance(dataset, dict) else dataset["train"][0]
            )

        columns = self["default"].column_names
        columns = self["default"].column_names

        if ("label" in columns and "text" in columns) or (
            "labels" in columns and "text" in columns
        ):
            task_categories = ["text-classification"]
        elif ("prompt" in columns and "completion" in columns) or (
            "messages" in columns
        ):
            task_categories: list[str] = [
                "text-generation",
                "text2text-generation",
                "question-answering",
            ]
        elif "context" in columns and "question" in columns and "response" in columns:
            task_categories: list[str] = [
                "text-generation",
                "text2text-generation",
                "text-retrieval",
                "question-answering"
            ]
            if (
                "positive_retrieval" in columns and "negative_retrieval" in columns
            ) or ("positive_reranking" in columns and "negative_reranking" in columns):
                task_categories.append("sentence-similarity")
        else:
            task_categories: list[str] = []
            gr.Info(
                f"No task categories found for dataset with columns: {columns}. "
                "Please notify the distilabel team if you think this is an error."
            )

        readme_metadata = {}
        if repo_id and token:
            readme_metadata = self._extract_readme_metadata(repo_id, token)

        metadata = {
            **readme_metadata,
            "size_categories": size_categories_parser(
                max(len(dataset) for dataset in self.values())
            ),
            "task_categories": task_categories,
            "tags": [
                "synthetic",
                "distilabel",
                "rlaif",
                "datacraft",
            ],
        }

        card = DistilabelDatasetCard.from_template(
            card_data=DatasetCardData(**metadata),
            repo_id=repo_id,
            sample_records=sample_records,
            include_script=include_script,
            filename_py=filename_py,
            references=self.citations,
        )

        return card


distilabel.distiset.Distiset = CustomDistisetWithAdditionalTag
