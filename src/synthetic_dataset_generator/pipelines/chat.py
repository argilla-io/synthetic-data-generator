from datasets import get_dataset_config_names, get_dataset_split_names
from distilabel.steps.tasks import (
    ChatGeneration,
    Magpie,
    GenerateSentencePair,
    TextGeneration,
)

from synthetic_dataset_generator.constants import (
    MAGPIE_PRE_QUERY_TEMPLATE,
    MAX_NUM_TOKENS,
)
from synthetic_dataset_generator.pipelines.base import _get_llm, _get_llm_class

INFORMATION_SEEKING_PROMPT = (
    "You are an AI assistant designed to provide accurate and concise information on a wide"
    " range of topics. Your purpose is to assist users in finding specific facts,"
    " explanations, or details about various subjects. Provide clear, factual responses and,"
    " when appropriate, offer additional context or related information that might be useful"
    " to the user."
)

REASONING_PROMPT = (
    "You are an AI assistant specialized in logical thinking and problem-solving. Your"
    " purpose is to help users work through complex ideas, analyze situations, and draw"
    " conclusions based on given information. Approach each query with structured thinking,"
    " break down problems into manageable parts, and guide users through the reasoning"
    " process step-by-step."
)

PLANNING_PROMPT = (
    "You are an AI assistant focused on helping users create effective plans and strategies."
    " Your purpose is to assist in organizing thoughts, setting goals, and developing"
    " actionable steps for various projects or activities. Offer structured approaches,"
    " consider potential challenges, and provide tips for efficient execution of plans."
)

EDITING_PROMPT = (
    "You are an AI assistant specialized in editing and improving written content. Your"
    " purpose is to help users refine their writing by offering suggestions for grammar,"
    " style, clarity, and overall structure. Provide constructive feedback, explain your"
    " edits, and offer alternative phrasings when appropriate."
)

CODING_DEBUGGING_PROMPT = (
    "You are an AI assistant designed to help with programming tasks. Your purpose is to"
    " assist users in writing, reviewing, and debugging code across various programming"
    " languages. Provide clear explanations, offer best practices, and help troubleshoot"
    " issues. When appropriate, suggest optimizations or alternative approaches to coding"
    " problems."
)

MATH_SYSTEM_PROMPT = (
    "You are an AI assistant designed to provide helpful, step-by-step guidance on solving"
    " math problems. The user will ask you a wide range of complex mathematical questions."
    " Your purpose is to assist users in understanding mathematical concepts, working through"
    " equations, and arriving at the correct solutions."
)

ROLE_PLAYING_PROMPT = (
    "You are an AI assistant capable of engaging in various role-playing scenarios. Your"
    " purpose is to adopt different personas or characters as requested by the user. Maintain"
    " consistency with the chosen role, respond in character, and help create immersive and"
    " interactive experiences for the user."
)

DATA_ANALYSIS_PROMPT = (
    "You are an AI assistant specialized in data analysis and interpretation. Your purpose is"
    " to help users understand and derive insights from data sets, statistics, and analytical"
    " tasks. Offer clear explanations of data trends, assist with statistical calculations,"
    " and provide guidance on data visualization and interpretation techniques."
)

CREATIVE_WRITING_PROMPT = (
    "You are an AI assistant designed to support creative writing endeavors. Your purpose is"
    " to help users craft engaging stories, poems, and other creative texts. Offer"
    " suggestions for plot development, character creation, dialogue writing, and other"
    " aspects of creative composition. Provide constructive feedback and inspire creativity."
)

ADVICE_SEEKING_PROMPT = (
    "You are an AI assistant focused on providing thoughtful advice and guidance. Your"
    " purpose is to help users navigate various personal or professional issues by offering"
    " balanced perspectives, considering potential outcomes, and suggesting practical"
    " solutions. Encourage users to think critically about their situations while providing"
    " supportive and constructive advice."
)

BRAINSTORMING_PROMPT = (
    "You are an AI assistant specialized in generating ideas and facilitating creative"
    " thinking. Your purpose is to help users explore possibilities, think outside the box,"
    " and develop innovative concepts. Encourage free-flowing thoughts, offer diverse"
    " perspectives, and help users build upon and refine their ideas."
)

PROMPT_CREATION_PROMPT = f"""You are an AI assistant specialized in generating very precise prompts for dataset creation.

Your task is to write a prompt following the instruction of the user. Respond with the prompt and nothing else.

In the generated prompt always finish with this sentence: User questions are direct and concise.

The prompt you write should follow the same style and structure as the following example prompts:

{INFORMATION_SEEKING_PROMPT}

{REASONING_PROMPT}

{PLANNING_PROMPT}

{CODING_DEBUGGING_PROMPT}

{EDITING_PROMPT}

{ROLE_PLAYING_PROMPT}

{DATA_ANALYSIS_PROMPT}

{CREATIVE_WRITING_PROMPT}

{ADVICE_SEEKING_PROMPT}

{BRAINSTORMING_PROMPT}

User dataset description:
"""

FOLLOW_UP_TEMPLATE = """Conversation:
{% for message in messages %}
    {% if message.role == "user" %}
User Question: {{ message.content }}
    {% elif message.role == "assistant" %}
Assistant Response: {{ message.content }}
    {% endif %}
{% endfor %}

Please generate the next logical user message in this conversation. Do not include any other information or 'User Question' in your response.
""".rstrip()

DEFAULT_DATASET_DESCRIPTIONS = [
    "rude customer assistant for a phone company",
    "assistant that solves math puzzles using python",
]
if MAGPIE_PRE_QUERY_TEMPLATE == "llama3":
    _STOP_SEQUENCES = [
        "<|eot_id|>",
        "<|start_header_id|>",
        "assistant",
        " \n\n",
    ]
elif MAGPIE_PRE_QUERY_TEMPLATE == "qwen2":
    _STOP_SEQUENCES = ["<|im_end|>", "<|im_start|>", "assistant", "\n\n"]
else:
    _STOP_SEQUENCES = [
        "<|eot_id|>",
        "<|start_header_id|>",
        "assistant",
        " \n\n",
    ]


def _get_output_mappings(num_turns: int):
    if num_turns == 1:
        return {"instruction": "prompt", "response": "completion"}
    else:
        return {"conversation": "messages"}


def get_prompt_generator():
    generation_kwargs = {
        "temperature": 0.8,
        "max_new_tokens": MAX_NUM_TOKENS,
        "do_sample": True,
    }
    prompt_generator = TextGeneration(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        system_prompt=PROMPT_CREATION_PROMPT,
        use_system_prompt=True,
    )
    prompt_generator.load()
    return prompt_generator


def get_magpie_generator(num_turns: int, temperature: float, is_sample: bool):
    input_mappings = _get_output_mappings(num_turns)
    output_mappings = input_mappings.copy()
    if num_turns == 1:
        generation_kwargs = {
            "temperature": temperature,
            "do_sample": True,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.25),
            "stop_sequences": _STOP_SEQUENCES,
        }
        magpie_generator = Magpie(
            llm=_get_llm(
                generation_kwargs=generation_kwargs,
                magpie_pre_query_template=MAGPIE_PRE_QUERY_TEMPLATE,
                use_magpie_template=True,
            ),
            n_turns=num_turns,
            output_mappings=output_mappings,
            only_instruction=True,
        )
    else:
        generation_kwargs = {
            "temperature": temperature,
            "do_sample": True,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.5),
            "stop_sequences": _STOP_SEQUENCES,
        }
        magpie_generator = Magpie(
            llm=_get_llm(
                generation_kwargs=generation_kwargs,
                magpie_pre_query_template=MAGPIE_PRE_QUERY_TEMPLATE,
                use_magpie_template=True,
            ),
            end_with_user=True,
            n_turns=num_turns,
            output_mappings=output_mappings,
        )
    magpie_generator.load()
    return magpie_generator


def get_sentence_pair_generator(temperature: float, is_sample: bool):
    generation_kwargs = {
        "temperature": temperature,
        "max_new_tokens": 256 if is_sample else MAX_NUM_TOKENS,
    }
    sentence_pair_generator = GenerateSentencePair(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        triplet=False,
        action="query",
        hard_negative=True,
    )
    sentence_pair_generator.load()
    return sentence_pair_generator


def get_response_generator(
    system_prompt: str, num_turns: int, temperature: float, is_sample: bool
):
    if num_turns == 1:
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.5),
        }
        response_generator = TextGeneration(
            llm=_get_llm(generation_kwargs=generation_kwargs),
            system_prompt=system_prompt,
            output_mappings={"generation": "completion"},
            input_mappings={"instruction": "prompt"},
        )
    else:
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": MAX_NUM_TOKENS,
        }
        response_generator = ChatGeneration(
            llm=_get_llm(generation_kwargs=generation_kwargs),
            output_mappings={"generation": "completion"},
            input_mappings={"conversation": "messages"},
        )
    response_generator.load()
    return response_generator


def get_follow_up_generator(type: str, temperature: float, is_sample: bool):
    if type == "instruction":
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": 256 if is_sample else int(MAX_NUM_TOKENS * 0.5),
        }
        follow_up_generator = TextGeneration(
            llm=_get_llm(generation_kwargs=generation_kwargs),
            template=FOLLOW_UP_TEMPLATE,
            columns=["messages"],
        )
    else:
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": MAX_NUM_TOKENS,
        }
        follow_up_generator = ChatGeneration(
            llm=_get_llm(generation_kwargs=generation_kwargs),
        )
    follow_up_generator.load()
    return follow_up_generator

def generate_pipeline_code_system_prompt(
    repo_id: str,
    input_type: str,
    system_prompt: str,
    document_column: str,
    num_turns: int,
    num_rows: int,
):
    input_mappings = _get_output_mappings(num_turns)
    code = f"""
    # Requirements: `pip install distilabel[hf-inference-endpoints]`
    import os
    from distilabel.pipeline import Pipeline
    from distilabel.steps import KeepColumns
    from distilabel.steps.tasks import MagpieGenerator
    from distilabel.llms import {_get_llm_class()}

    SYSTEM_PROMPT = "{system_prompt}"

    with Pipeline(name="sft") as pipeline:
        magpie = MagpieGenerator(
            llm={_get_llm_class()}.from_dict(
                {_get_llm().dump()}
            ),
            n_turns={num_turns},
            num_rows={num_rows},
            batch_size=1,
            system_prompt=SYSTEM_PROMPT,
            output_mappings={input_mappings},
        )
        keep_columns = KeepColumns(
            columns={list(input_mappings.values())} + ["model_name"],
        )
        magpie.connect(keep_columns)

    if __name__ == "__main__":
        distiset = pipeline.run()
    """
    return code

def generate_pipeline_code_seed(
    repo_id: str,
    subset: str,
    split: str,
    input_type: str,
    system_prompt: str,
    document_column: str,
    num_turns: int,
    num_rows: int,
):
    code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
from distilabel.models import {_get_llm_class()}
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns{", LoadDataFromDicts" if input_type != "dataset-input"  else ""}{", LoadDataFromHub" if input_type == "dataset-input" else ""}
from distilabel.steps.tasks import GenerateSentencePair, TextGeneration {", ChatGeneration" if num_turns > 1 else ""}
"""

    if num_turns > 1:
        code += """
FOLLOW_UP_TEMPLATE = '''Conversation:
{{% for message in messages %}}
    {{% if message.role == "user" %}}
User Question: {{{{ message.content }}}}
    {{% elif message.role == "assistant" %}}
Assistant Response: {{{{ message.content }}}}
    {{% endif %}}
{{% endfor %}}

Please generate the next logical user message in this conversation. Do not include any other information or 'User Question' in your response.
'''.rstrip()

@step(inputs=["prompt", "completion"], outputs=["messages"])
def PrepareMessages(*inputs: StepInput) -> StepOutput:
    for input in inputs:
        for item in input:
            item["messages"] = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]
        yield input


@step(inputs=["messages", "generation"], outputs=["messages"])
def FormatMessagesInstruction(*inputs: StepInput) -> StepOutput:
    for input in inputs:
        for item in input:
            item["messages"].append({"role": "user", "content": item["generation"]})
        yield input


@step(inputs=["messages", "generation"], outputs=["messages"])
def FormatMessagesResponse(*inputs: StepInput) -> StepOutput:
    for input in inputs:
        for item in input:
            item["messages"].append({"role": "assistant", "content": item["generation"]})
        yield input
"""

    if input_type == "dataset-input":
        code += f"""
with Pipeline(name="sft") as pipeline:
    load_the_dataset = LoadDataFromHub(
        repo_id='{repo_id}',
        config='{subset}',
        split='{split}',
        num_examples={num_rows},
        batch_size=2,
        output_mappings={{'{document_column}':'anchor'}},
    )
    """

    else: 
        code += """
data = process_and_chunk_files(files=[files])

with Pipeline(name="sft") as pipeline:
    load_the_dataset = LoadDataFromDicts(
        data = data,
    )
"""
    code += f"""
    instruction_generator = GenerateSentencePair(
        name="instruction_generation",
        triplet=False,
        hard_negative=True,
        action="query",
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
        input_batch_size=10,
        output_mappings={{"positive": "prompt"}},
    )

    response_generator = TextGeneration(
        name="response_generation",
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
        input_batch_size=10,
        input_mappings={{"instruction": "prompt"}},
        output_mappings={{"generation": "completion"}},
    )
    """

    if num_turns > 1:
        code += """
    prepare_messages = PrepareMessages()
    """

        for i in range(num_turns - 1):
            code += f"""
    follow_up_instruction_{i} = TextGeneration(
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
        template=FOLLOW_UP_TEMPLATE,
        columns=["messages"],
    )
    format_instruction_{i} = FormatMessagesInstruction()
    follow_up_response_{i} = ChatGeneration(
        llm={_get_llm_class()}.from_dict(
            {_get_llm().dump()}
        ),
    )
    format_response_{i} = FormatMessagesResponse()
    """

    if num_turns > 1:
        code += """
        keep_columns = KeepColumns(columns=["messages"])
        """
        code += "load_the_dataset >> instruction_generator >> response_generator >> prepare_messages"
        
        for i in range(1, num_turns + 1):
            code += f" >> follow_up_instruction_{i} >> format_instruction_{i} >> follow_up_response_{i} >> format_response_{i}"
            
        code += " >> keep_columns"

    code += """
if __name__ == "__main__":
    distiset = pipeline.run()
)
"""
    return code

def generate_pipeline_code(
    repo_id: str,
    input_type: str,
    system_prompt: str,
    document_column: str,
    num_turns: int,
    num_rows: int,
):
    if input_type == "dataset-input" and repo_id is not None:
        subset = get_dataset_config_names(repo_id)[0]
        split = get_dataset_split_names(repo_id, subset)[0]
    else:
        subset = "default"
        split = "train"
    if input_type == "prompt-type":
        return generate_pipeline_code_system_prompt(
            system_prompt=system_prompt,
            num_turns=num_turns,
            num_rows=num_rows,
        )
    return generate_pipeline_code_seed(
        repo_id=repo_id,
        subset=subset,
        split=split,
        input_type=input_type,
        system_prompt=system_prompt,
        document_column=document_column,
        num_turns=num_turns,
        num_rows=num_rows,
    )
