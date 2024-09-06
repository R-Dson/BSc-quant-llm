

"""
This module contains functions for generating few-shot prompts for various models.
The main function is generate_few_shot_prompt which constructs a series of messages
representing the context, few-shot examples, and current question to be answered by the model.
Other helper functions format the questions and options based on the dataset name provided.
"""
def generate_few_shot_prompt(few_shot_dict: dict, question: dict, system: str, model_name: str, dataset_name: str):
    """
    Generates a few-shot prompt for the language model by constructing a series of messages
    based on the provided dictionary of examples, current question, and additional details.

    Args:
        few_shot_dict (dict): A dictionary containing multiple choice questions as values. If 'category' is in `question`,
                                then the function will filter `few_shot_dict` to only include those with the specified category.
        question (dict): A dictionary representing a single multiple choice question, which may contain fields such as "category".
        system (str): The initial system message or instruction for the language model.
        model_name (str): Name of the model to generate prompt for. GEMMA model does not support system prompt, so it is handled separately.
        dataset_name (str): Name of the dataset used for context and formatting question-answer pairs.

    Returns:
        list: A list of dictionaries representing messages in a conversation format, with each dictionary having "role" and "content" fields.
    """
    # Filter few-shot examples based on 'category' if it exists in the current question
    if 'category' in question:
        subject = question["category"]
        system += f"\nThe following are multiple choice questions (with answers) about {subject}."
        few_shot_dict = few_shot_dict[question['category']]

    # Initialize messages with system prompt, if supported by the model
    if 'gemma' not in model_name: # gemma does not support system prompt
        messages = [{"role": "system", "content": system}]
    else: 
        messages = [{"role": "user", "content": system}, {"role": "assistant", "content": "Understood. I will now follow the instructions carefully, never ask for clarification, never ask questions or follow-up questions, and never offer further assistance."}]
    
    # Add few-shot examples to messages
    for q in few_shot_dict:
        content = generate_cot_prompt(q, dataset_name)
        
        messages.append({'role': 'user', 'content': content})
        if 'cot_content' in q:
            cot_content = q['cot_content']
        else:
            # Format answer based on the dataset name
            if dataset_name == 'TIGER-Lab/MMLU-Pro':
                cot_content = f'The answer is ({q['output']}).'
            elif dataset_name == 'cruxeval-org/cruxeval':
                cot_content = f'The output is {q["output"]}.'
            elif dataset_name == 'TAUR-Lab/MuSR':
                letter = chr(65 + (q['answer_index'] % 26))
                cot_content = f'The answer is ({letter}).'
                
        cot_content = cot_content.replace('A: ', '')
        messages.append({"role": "assistant", "content": cot_content})
    
    return messages
    
def format_cot(example, dataset_name, including_answer=True) -> str:
    """
    Formats a multiple-choice question and its options based on the provided dataset name.

    Args:
        example (dict): A dictionary representing a single multiple choice question with fields such as "question", "options", etc.
        dataset_name (str): Name of the dataset used for formatting question-answer pairs.

    Returns:
        str: A string containing the formatted question and options.
    """
    # Format question and options based on the dataset name

    if dataset_name == 'TIGER-Lab/MMLU-Pro':
        question = example["question"]
        options = example["options"]
        prompt = "Question:\n"
        prompt += question + "\n"
        prompt += "Options:\n"
        for i, opt in enumerate(options):
            prompt += f"{chr(65+i)}. {opt}\n"  # Use A, B, C, ... as choices
    elif dataset_name == 'cruxeval-org/cruxeval':
        prompt = "Code:\n"
        code = example['code']
        inp = example['input']
        prompt = f"""Code:
{code}

Input:

{inp}"""
    elif dataset_name == 'TAUR-Lab/MuSR':
        question = example["question"]
        narrative = example["narrative"]
        options = example['choices']
        actual_list = options.strip("[]").replace("'", "").split(", ")

        prompt = "Narrative:\n"
        prompt += narrative + "\n\n"
        prompt += "Question:\n"
        prompt += question + "\n\n"
        prompt += "Options:\n"
        for i, opt in enumerate(actual_list):
            prompt += f"{chr(65+i)}. {opt}\n"  # Use A, B, C, ... as choices
        
    return prompt

"""
Args:
    curr (dict): A dictionary representing a single multiple choice question with fields such as "question", "options", etc.
    dataset_name (str): Name of the dataset used for formatting question-answer pairs.

Returns:
    str: A string containing the formatted prompt for the current question.
"""
def generate_cot_prompt(curr, dataset_name) -> str:
# Format prompt based on the dataset name

    """if 'category' in curr:
        subject = curr["category"]
    else:
        subject = 'eval'"""
    prompt = format_cot(curr, dataset_name)
    return prompt
