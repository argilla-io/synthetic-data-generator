from datasets import get_dataset_config_names, get_dataset_split_names
from distilabel.steps.tasks import (
    GenerateSentencePair,
    TextGeneration,
)

from synthetic_dataset_generator.constants import MAX_NUM_TOKENS
from synthetic_dataset_generator.pipelines.base import _get_llm, _get_llm_class

DEFAULT_DATASET_DESCRIPTIONS = [
    "A dataset to retrieve information from legal documents.",
    "A dataset to search for economical techniques.",
]

PROMPT_CREATION_PROMPT = """

You are an AI assistant specialized in designing retrieval-augmented generation (RAG) tasks for dataset generation.

Your task is to generate a well-structured and descriptive prompt based on the provided dataset description. Respond with only the generated prompt and nothing else.

The prompt should closely follow the style and structure of the example prompts below. Ensure that you include all relevant details from the dataset description.

Description: A dataset to retrieve information from legal documents.
Output: A dataset to retrieve information from a collection of legal documents related to the US law system and the status of contracts.

Description: A dataset to search for economical techniques.
Output: A dataset to search for economical techniques and strategies for the European market and the financial sector.

Description: A dataset covering FAQ questions for a tech company called Argilla that sells technology datasets within the open-source Natural Language Processing space.
Output: A dataset covering FAQ questions for a tech company called Argilla that sells technology datasets within the open-source Natural Language Processing space.

Description:
"""

SYSTEM_PROMPT_CHUCKS = """
You are a helpful and knowledgeable AI assistant. Your task is to generate concise and informative text chunks relevant to the given retrieval task.

Ensure the text chunks are:
- Focused and directly related to the retrieval task.
- Clear, truthful, and based on your general knowledge.

Do not include or reference the retrieval task itself in the generated chunks.
"""

CHUNKS_TEMPLATE = """You have been assigned to generate text chunks based on the following retrieval task: {{ task }}.

Provide only the text chunks without explaining your process or reasoning. Do not include any additional information. Do not indicate that it is a text chunk.

Ensure the chunks are concise, clear, and directly relevant to the task.

Use your general knowledge to create informative and precise outputs.
"""

SYSTEM_PROMPT_RAG = """
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
    text_generator = TextGeneration(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        system_prompt=PROMPT_CREATION_PROMPT,
        use_system_prompt=True,
    )

    text_generator.load()
    return text_generator


def get_chunks_generator(temperature: float, is_sample: bool):
    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": MAX_NUM_TOKENS if is_sample else 256,
    }
    text_generator = TextGeneration(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        system_prompt=SYSTEM_PROMPT_CHUCKS,
        template=CHUNKS_TEMPLATE,
        columns=["task"],
        use_system_prompt=True,
    )

    text_generator.load()
    return text_generator


def get_sentence_pair_generator(action: str, triplet: bool, temperature: float, is_sample: bool):
    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": 256 if is_sample else MAX_NUM_TOKENS,
    }
    sentence_pair_generator = GenerateSentencePair(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        triplet=triplet,
        action=action,
        hard_negative=True,
    )
    sentence_pair_generator.load()
    return sentence_pair_generator


def get_response_generator(temperature: float, is_sample: bool):
    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": MAX_NUM_TOKENS if is_sample else 256,
    }
    text_generator = TextGeneration(
        llm=_get_llm(is_completion=True, generation_kwargs=generation_kwargs),
        system_prompt=SYSTEM_PROMPT_RAG,
        template=RAG_TEMPLATE,
        columns=["context", "question"],
        use_system_prompt=True,
    )

    text_generator.load()
    return text_generator


def generate_pipeline_code(
    repo_id: str,
    input_type: str,
    system_prompt: str,
    document_column: str,
    retrieval_reranking: list[str],
    num_rows: int = 10,
) -> str:
    if input_type == "dataset-input" and repo_id is not None:
        subset = get_dataset_config_names(repo_id)[0]
        split = get_dataset_split_names(repo_id, subset)[0]
    else:
        subset = "default"
        split = "train"
    retrieval = "Retrieval" in retrieval_reranking
    reranking = "Reranking" in retrieval_reranking
    base_code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
{"import random" if input_type == "prompt-input" else ""}
from distilabel.models import {_get_llm_class()}
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns{", LoadDataFromDicts" if input_type != "dataset-input"  else ""}{", LoadDataFromHub" if input_type == "dataset-input" else ""}{", CombineOutputs" if retrieval and reranking else ""}
from distilabel.steps.tasks import GenerateSentencePair, TextGeneration {", GenerateTextRetrievalData" if input_type == "prompt-input" else ""}

SYSTEM_PROMPT_RAG = '''
You are a helpful AI assistant. Your task is to answer the following question based on the provided document.

If the answer is not explicitly stated in the document, use your knowledge to provide the most relevant and accurate answer possible.

If you cannot answer the question based on the given information, state that clearly.
'''

RAG_TEMPLATE = '''Document:
{{{{ filename }}}}

Question: {{{{ question }}}}

Please provide a clear and concise answer to the question based on the information in the document:
'''.rstrip()
"""

    if input_type == "file-input":
        base_code += """
data = process_and_chunk_files(files=[files])
"""

    if input_type == "prompt-input":
        pipeline = f"""
TASK_SYSTEM_PROMPT =  '''

{system_prompt}    
''' 

with Pipeline(name="rag") as pipeline:

    task_generator = LoadDataFromDicts(data=[{{"task": TASK_SYSTEM_PROMPT}}])

    sentence_similarity_generation = GenerateTextRetrievalData(
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
        seed=random.randint(0, 2**32 - 1),
        query_type="common",
        difficulty="high school",
        clarity="clear",
        num_generations={num_rows},
        output_mappings={{"positive_document": "anchor"}},
    )

    keep_columns_prompt = KeepColumns(
        columns=["anchor"],
    )
    """
    else:
        pipeline = """
with Pipeline(name="rag") as pipeline:
"""
        if input_type == "file-input":
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
        batch_size=2,
        output_mappings={{'{document_column}': 'anchor'}}
    )
    """

    pipeline += f"""
    generate_retrieval_pairs = GenerateSentencePair(
        triplet={str(retrieval)},
        hard_negative=True,
        action="query",
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
        output_mappings={{"positive": "positive_retrieval"{', "negative": "negative_retrieval"' if retrieval else ""}}},
        input_batch_size=10,
    )
    """

    if reranking:
        pipeline += f"""
    generate_reranking_pairs = GenerateSentencePair(
        triplet=True,
        hard_negative=True,
        action="semantically-similar",
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
        input_batch_size=10,
        output_mappings={{"positive": "positive_reranking", "negative": "negative_reranking"}},
    )
    
    combine_outputs = CombineOutputs()
    """

    pipeline += f"""
    generate_response = TextGeneration(
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
        system_prompt=SYSTEM_PROMPT_RAG,
        template=RAG_TEMPLATE,
        columns=["filename", "question"],
        use_system_prompt=True,
        input_mappings={{"filename": "anchor", "question": "positive_retrieval"}},
        output_mappings={{"generation": "response"}},
    )
    
    keep_columns = KeepColumns(
        columns=["anchor", "positive_retrieval", "response"{', "negative_retrieval"' if retrieval else ""}{', "positive_reranking", "negative_reranking"' if reranking else ""}],
    )
    """

    pipeline_steps = (
        "[generate_retrieval_pairs, generate_reranking_pairs] >> combine_outputs >> generate_response >> keep_columns"
        if reranking
        else "generate_retrieval_pairs >> generate_response >> keep_columns"
    )

    pipeline += """
    task_generator >> sentence_similarity_generation >> keep_columns_prompt >> {pipeline_steps}
""".format(pipeline_steps=pipeline_steps) if input_type == "prompt-input" else """
    load_the_dataset >> {pipeline_steps}
""".format(pipeline_steps=pipeline_steps)

    pipeline += """
    if __name__ == "__main__":
        distiset = pipeline.run()
    """

    return base_code + pipeline
