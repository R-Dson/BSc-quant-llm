import os
import subprocess
from datasets import load_dataset
import re
import json

def save_perplexity(model_name: str, ppl: list, quant: str = ''):
    """
    Save the perplexity results of a model to a JSON file.

    Args:
        model_name (str): The name of the model.
        ppl (list): A list containing the perplexity results.
        quant (str, optional): The quantization type used for the model. Defaults to ''.
    """
    # Create a directory for storing results if it doesn't exist
    if not os.path.exists('results'):
        os.mkdir('results')
    # Create a subdirectory for the specific model if it doesn't exist
    if not os.path.exists(f'results/{model_name}'):
        os.mkdir(f'results/{model_name}')
    # Add quantization type to filename if provided
    if quant != '':
        quant = f'-{quant}'
    # Save perplexity results to a JSON file
    with open(f'results/{model_name}/{model_name}{quant}-PPL.json', 'w+') as f:
        json.dump(ppl, f, indent=4, separators=(',', ': '))
    
    print(f'saved ppl to: results/{model_name}/{model_name}-PPL.json')
    pass

def perplexity(full_model_path: str):
    """
    Calculate the perplexity of a language model on the Wikitext-2 test set.

    Args:
        full_model_path (str): The path to the language model file.

    Returns:
        tuple or None: A tuple containing the estimate and margin of error if successful, otherwise None.
    """
    # Check if model file exists
    if not os.path.exists(full_model_path):
        return
    
    # Download Wikitext-2 test set if it doesn't exist
    if not os.path.exists('./wikitext-2-raw'):
        in_string = f"./llama.cpp/scripts/get-wikitext-2.sh"
        process = subprocess.Popen(in_string, shell=True, stdout=subprocess.PIPE)
        # wait for the process to finish
        output, error = process.communicate()
        output = output.decode('utf-8')
        return_code = process.returncode
        if return_code == 0:
            print("Process completed successfully")
            # Print the output of the download script
            print("Output:", output)
        else:
            return

    # Set number of layers for perplexity calculation based on model size and quantization type
    ngl = -1
    if '70b' in full_model_path:
        if 'Q2' in full_model_path:
            ngl = 70
        elif 'Q4' in full_model_path:
            ngl = 42
        elif 'Q6' in full_model_path:
            ngl = 30
        elif 'Q8' in full_model_path:
            ngl = 24 
    
    # Calculate perplexity using llama.cpp/llama-perplexity script
    in_string = f"llama.cpp/llama-perplexity -m {full_model_path} -ngl {ngl} -t 16 -f wikitext-2-raw/wiki.test.raw"
    print(in_string)
    process = subprocess.Popen(in_string, shell=True, stdout=subprocess.PIPE)
    # wait for the process to finish
    output, error = process.communicate()
    output = output.decode('utf-8')
    return_code = process.returncode

    # Extract perplexity estimate and margin of error from script output using regex
    match = re.search(r'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)', output)
    
    if return_code == 0:
        print("Process completed successfully")
        # Return perplexity estimate and margin of error as a tuple
        if match:
            estimate = float(match.group(1))
            margin_of_error = float(match.group(2))
            return estimate, margin_of_error
    else:
        return None

def main():
    perplexity('')
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")




if __name__ == '__main__':
    main()
