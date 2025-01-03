from typing import List

from datasets import get_dataset_config_names, get_dataset_split_names
from distilabel.llms import InferenceEndpointsLLM, OpenAILLM
from distilabel.steps.tasks import (
    GenerateSentencePair,
    TextGeneration,
)

from synthetic_dataset_generator.constants import BASE_URL, MAX_NUM_TOKENS, MODEL
from synthetic_dataset_generator.pipelines.base import _get_next_api_key

DEFAULT_DATASET_DESCRIPTIONS = [
    "A dataset to retrieve information from legal documents.",
    "A dataset to search for economical techniques.",
]

PROMPT_CREATION_PROMPT = """"You are an AI assistant specialized in generating very precise retrieval augmented generation tasks for dataset creation.

Your should write a prompt following a the dataset description. Respond with the prompt and nothing else.

The prompt should follow the same style and structure as the following example prompts.

Make sure to always include all of the detailed information from the description and the context of the company that is provided.

Description: A dataset to retrieve information from legal documents.
Output: A dataset to retrieve information from a collection of legal documents related to the US law system and the status of contracts.

Description: A dataset to search for economical techniques.
Output: A dataset to search for economical techniques and strategies for the European market and the financial sector.

Description: A dataset covering FAQ questions for a tech company called Argilla that sells technology datasets within the open source Natural Language Processing space.
Output: A dataset covering FAQ questions for a tech company called Argilla that sells technology datasets within the open source Natural Language Processing space.

Description:
"""

SYSTEM_PROMPT_CHUCKS = """"
You are a helpful and knowledgeable AI assistant. Your task is to generate concise and informative text chunks relevant to the given retrieval task.

Ensure the text chunks are:
- Focused and directly related to the retrieval task.
- Clear, truthful, and based on your general knowledge.

Do not include or reference the retrieval task itself in the generated chunks.
"""

CHUNKS_TEMPLATE = """You have been assigned to generate text chunks based on the following retrieval task: {{ task }}.

Provide only the text chunks without explaining your process or reasoning.

Ensure the chunks are clear, accurate, and directly relevant to the task.

Use your general knowledge to create informative and precise outputs.
"""

SYSTEM_PROMPT_RAG = """"
You are a helpful AI assistant. Your task is to answer the following question based on the provided document.

If the answer is not explicitly stated in the document, use your knowledge to provide the most relevant and accurate answer possible.

If you cannot answer the question based on the given information, state that clearly.
"""

RAG_TEMPLATE = """Document:
{{ context }}

Question: {{ question }}

Please provide a clear and concise answer to the question based on the information in the document:
""".rstrip()


def get_prompt_generator():
    generation_kwargs = {
        "temperature": 0.8,
        "max_new_tokens": MAX_NUM_TOKENS,
    }
    if BASE_URL:
        llm = OpenAILLM(
            model=MODEL,
            base_url=BASE_URL,
            api_key=_get_next_api_key(),
            generation_kwargs=generation_kwargs,
        )
    else:
        generation_kwargs["do_sample"] = True
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            model_id=MODEL,
            base_url=BASE_URL,
            generation_kwargs=generation_kwargs,
        )

    text_generator = TextGeneration(
        llm=llm,
        system_prompt=PROMPT_CREATION_PROMPT,
        use_system_prompt=True,
    )

    text_generator.load()
    return text_generator


def get_chunks_generator(temperature, is_sample):
    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": MAX_NUM_TOKENS if is_sample else 256,
    }
    if BASE_URL:
        llm = OpenAILLM(
            model=MODEL,
            base_url=BASE_URL,
            api_key=_get_next_api_key(),
            generation_kwargs=generation_kwargs,
        )
    else:
        generation_kwargs["do_sample"] = True
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            model_id=MODEL,
            base_url=BASE_URL,
            generation_kwargs=generation_kwargs,
        )

    text_generator = TextGeneration(
        llm=llm,
        system_prompt=SYSTEM_PROMPT_CHUCKS,
        template=CHUNKS_TEMPLATE,
        columns=["task"],
        use_system_prompt=True,
    )

    text_generator.load()
    return text_generator


def get_sentence_pair_generator(action, triplet, hard_negative, temperature, is_sample):
    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": 256 if is_sample else MAX_NUM_TOKENS,
    }
    if BASE_URL:
        llm = OpenAILLM(
            model=MODEL,
            base_url=BASE_URL,
            api_key=_get_next_api_key(),
            generation_kwargs=generation_kwargs,
        )
    else:
        generation_kwargs["do_sample"] = True
        llm = InferenceEndpointsLLM(
            model_id=MODEL,
            base_url=BASE_URL,
            api_key=_get_next_api_key(),
            generation_kwargs=generation_kwargs,
        )

    sentence_pair_generator = GenerateSentencePair(
        llm=llm,
        triplet=triplet,
        action=action,
        hard_negative=hard_negative,
    )
    sentence_pair_generator.load()
    return sentence_pair_generator


def get_response_generator(temperature, is_sample):
    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": MAX_NUM_TOKENS if is_sample else 256,
    }
    if BASE_URL:
        llm = OpenAILLM(
            model=MODEL,
            base_url=BASE_URL,
            api_key=_get_next_api_key(),
            generation_kwargs=generation_kwargs,
        )
    else:
        generation_kwargs["do_sample"] = True
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            model_id=MODEL,
            base_url=BASE_URL,
            generation_kwargs=generation_kwargs,
        )

    text_generator = TextGeneration(
        llm=llm,
        system_prompt=SYSTEM_PROMPT_RAG,
        template=RAG_TEMPLATE,
        columns=["context", "question"],
        use_system_prompt=True,
    )

    text_generator.load()
    return text_generator


def generate_pipeline_code(
    repo_id: str,
    file_paths: List[str],
    input_type: str,
    document_column: str,
    hard_negative: bool = False,
    retrieval: bool = False,
    reranking: bool = False,
    num_rows: int = 10,
    temperature: float = 0.9,
) -> str:
    MODEL_ARG = "model_id" if BASE_URL else "model"
    MODEL_CLASS = "InferenceEndpointsLLM" if BASE_URL else "OpenAILLM"
    if repo_id is None:
        subset = "default"
        split = "train"
    else:
        subset = get_dataset_config_names(repo_id)[0]
        split = get_dataset_split_names(repo_id, subset)[0]
    base_code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
import os
import random
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns{", LoadDataFromDicts" if input_type == "file-type" else ""}{", LoadDataFromHub" if input_type == "dataset_input" else ""}
from distilabel.steps.tasks import GenerateSentencePair, TextGeneration {", GenerateTextRetrievalData" if input_type == "prompt-type" else ""}
MODEL = "{MODEL}"
BASE_URL = "{BASE_URL}"
SYSTEM_PROMPT_RAG = '''
You are a helpful AI assistant. Your task is to answer the following question based on the provided document.

If the answer is not explicitly stated in the document, use your knowledge to provide the most relevant and accurate answer possible.

If you cannot answer the question based on the given information, state that clearly.
'''

RAG_TEMPLATE = '''Document:
{{ {document_column} }}

Question: {{ question }}

Please provide a clear and concise answer to the question based on the information in the document:
'''.rstrip()

os.environ["API_KEY"] = (
    "hf_xxx"  # https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&canReadGatedRepos=true&tokenType=fineGrained
)
"""
    if input_type == "file_type":
        base_code += f"""
data = process_and_chunk_files(files=[{file_paths}])
"""

    if input_type == "prompt-type":
        pipeline = f"""
with Pipeline(name="textcat") as pipeline:

    task_generator = LoadDataFromDicts(data=[{{"task": TEXT_CLASSIFICATION_TASK}}])

    textcat_generation = GenerateTextClassificationData(
        llm={MODEL_CLASS}(
            {MODEL_ARG}=MODEL,
            base_url=BASE_URL,
            api_key=os.environ["API_KEY"],
            generation_kwargs={{
                "temperature": {temperature},
                "max_new_tokens": {MAX_NUM_TOKENS},
                "top_p": 0.95,
            }},
        ),
        seed=random.randint(0, 2**32 - 1),
        query_type="common",
        difficulty="high school",
        clarity="clear",
        num_generations={num_rows},
        output_mappings={{"positive_document": "anchor"}},
    )

    keep_columns = KeepColumns(
        columns=["anchor"],
    )
        """

    else:
        pipeline = """"
with Pipeline(name="rag") as pipeline:
"""
    if input_type == "file_type":
        pipeline += """
    load_the_dataset = LoadDataFromDicts(
        data = data,
    )
    """
    else:
        pipeline += f"""
    load_the_dataset = LoadDataFromHub(
        repo_id="{repo_id}",
        config="{subset}",
        split="{split}",
        num_examples={num_rows},
        batch_size=2
    )
        """

    pipeline += f"""
    generate_retrieval_pairs = GenerateSentencePair(
        triplet={True if retrieval else False},
        hard_negative={hard_negative},
        action="query",
        llm={MODEL_CLASS}(
            {MODEL_ARG}=MODEL,
            base_url=BASE_URL,
            api_key=os.environ["API_KEY"],
            generation_kwargs={{
                "temperature": {temperature},
                "max_new_tokens": {MAX_NUM_TOKENS},
            }},
        ),
        output_mappings={{"positive": "positive_retrieval", "negative": "negative_retrieval"}},
        input_batch_size=10,
    )

    generate_response = TextGeneration(
        llm={MODEL_CLASS}(
            {MODEL_ARG}=MODEL,
            base_url=BASE_URL,
            api_key=os.environ["API_KEY"],
            generation_kwargs={{
                "temperature": {temperature},
                "max_new_tokens": {MAX_NUM_TOKENS},
            }},
        ),
        system_prompt=SYSTEM_PROMPT,
        template=RAG_TEMPLATE,
        columns=["{document_column}", "question"],
        use_system_prompt=True,
        input_mappings={{"question": "positive_retrieval"}},
        output_mappings={{"generation": "response"}},
    )
    """

    if reranking:
        pipeline += f"""
    generate_reranking_pairs = GenerateSentencePair(
        triplet=True,
        hard_negative={hard_negative}
        action="semantically-similar",
        llm={MODEL_CLASS}(
            {MODEL_ARG}=MODEL,
            base_url=BASE_URL,
            api_key=os.environ["API_KEY"],
            generation_kwargs={{
                "temperature": {temperature},
                "max_new_tokens": {MAX_NUM_TOKENS},
            }},
        ),
        input_batch_size=10,
        output_mappings={{"positive": "positive_reranking", "negative": "negative_reranking"}},
    )
    """
    # TODO: add https://distilabel.argilla.io/dev/components-gallery/steps/combineoutputs/ when released

    if input_type == "prompt-type":
        if reranking:
            pipeline += """
    task_generator.connect(textcat_generation)
    textcat_generation.connect(keep_columns)
    keep_columns.connect(generate_retrieval_pairs, generate_reranking_pairs)
    generate_retrieval_pairs.connect(generate_response)
    """
        else:
            pipeline += """
    task_generator.connect(textcat_generation)
    textcat_generation.connect(keep_columns)
    keep_columns.connect(generate_retrieval_pairs)
    generate_retrieval_pairs.connect(generate_response)
    """

    else:
        if reranking:
            pipeline += """
    load_dataset.connect(generate_retrieval_pairs, generate_reranking_pairs)
    generate_retrieval_pairs.connect(generate_response)
    """
        else:
            pipeline += """
    load_dataset.connect(generate_retrieval_pairs)
    generate_retrieval_pairs.connect(generate_response)
    """

    return base_code + pipeline
