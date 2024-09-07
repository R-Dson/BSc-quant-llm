from llama_cpp import Llama
import time

from together.types import ChatCompletionResponse

def llama_cpp_call_messages(llm: Llama, few_shot_messages: list, dataset_settings: dict):
    """
    This function calls the LLaMA model to generate a response based on a set of messages and settings.

    Parameters:
        llm (Llama): An instance of the Llama class representing the language model.
        few_shot_messages (list): A list of messages that provide context for the AI's response.
        dataset_settings (dict): A dictionary containing settings for the AI's response, such as maximum tokens and temperature.

    Returns:
        tuple: A tuple containing the generated response, the number of tokens per second, the time taken in milliseconds, and the total number of tokens used.
    """

    # Record the start time to calculate the duration of the API call later
    start_time = time.time()
    # Define a list of stop sequences that the AI should not include in its response
    stop = ["Question:", "</s>", "<end_of_turn>", "<|end|>", "<|endoftext|>" ]
    if "Q2" in llm.model_path:
         stop.append('\n') 

    # Call the AI's create_chat_completion method with the provided messages, settings, and stop sequences
    response = llm.create_chat_completion(messages=few_shot_messages, 
                                          max_tokens=dataset_settings['max_tokens'], 
                                          temperature=dataset_settings['temperature'], 
                                          #repeat_penalty=1.5,
                                          stop=stop
                                          )
                                    
    # Calculate the duration of the API call in seconds and milliseconds      
    delta_time = time.time() - start_time
    delta_ms = round(delta_time * 1000**2) / 1000
    try:
        # Extract usage information from the response, such as the total number of tokens used
        out_tokens = response['usage']
        total_tokens = out_tokens['total_tokens']
        # Extract the generated response from the response object
        response = response['choices'][0]['message']['content'].strip()
    except:
        # If an error occurs while extracting information from the response, use this alternate method
        total_tokens = response.usage.total_tokens
        response = response.choices[0].message.content.strip()
    
    # Calculate the number of tokens per second based on the total number of tokens and the duration of the API call
    tok_per_s = total_tokens * 1000 / delta_ms
    # Remove any trailing newline characters or periods from the generated response
    if len(response) > 0 and (response[-2:-1] == "\n" or response[-1] == "."):
            response = response[:-1]
        
    # Return the generated response, tokens per second, duration in milliseconds, and total number of tokens used
    return response, tok_per_s, delta_ms, total_tokens



"""
def llama_cpp_call_old(llm: Llama, prompt: str, model_name: str, SYSTEM: str) -> str:
    
    Calls a language model (LLM) with the provided prompt and returns its response.
    
    max_tokens = 2048
    temp = 0.1
    # For gemma
    # -e --temp 0 --repeat-penalty 1.0 --no-penalize-nl
    if 'gemma' in model_name:
        inputs = f""""""[SYSTEM]: {SYSTEM}\n\n{prompt}""""""
        output = llm(inputs, 
                     max_tokens=max_tokens,
                     temperature=temp,
                     repeat_penalty=1.0
                     )
        response = output['choices'][0]['text'].strip()
    else:
        output = llm.create_chat_completion(messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ], max_tokens=max_tokens,
        temperature=temp,
        repeat_penalty=1.0)
        response = output['choices'][0]['message']['content'].strip()

    if response[-4:-1] == "\n\n" :
        response = response[:-4]
    if response[-2:-1] == "\n":             
        response = response[:-2]
    if response[-1] == ".":
        response = response[:-1]
    return response
"""