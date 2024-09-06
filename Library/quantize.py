import os
import subprocess

def quantize_to_f16(model, full_model_path, out_path):
    """
    This function takes as input a model name, the path to its full version, and an output path.
    It uses the convert-hf-to-gguf.py script from llama.cpp to convert the given model from Hugging Face format to f16 gguf format.
    The converted model is saved in the specified output path with a filename that includes the model name and '-f16' suffix.
    After saving the model, it prints a message indicating that the quantization process has finished.

    Args:
        model (str): The name of the model to be quantized.
        full_model_path (str): The path to the full version of the model.
        out_path (str): The path where the converted model will be saved.
    """
    out_file = f'{out_path}{model}-bf16.gguf'

    if os.path.exists(out_file):
        return

    in_string = f"python llama.cpp/convert_hf_to_gguf.py {full_model_path} --outtype bf16 --outfile {out_file}"
    print(in_string)
    process = subprocess.Popen(in_string, shell=True, stdout=subprocess.PIPE)
    # wait for the process to finish
    output, error = process.communicate()
    print(f'Finished quantizing {model} to {out_file}')

def quantize_to_f32(model, full_model_path, out_path):
    """
    This function takes as input a model name, the path to its full version, and an output path.
    It uses the convert-hf-to-gguf.py script from llama.cpp to convert the given model from Hugging Face format to f16 gguf format.
    The converted model is saved in the specified output path with a filename that includes the model name and '-f16' suffix.
    After saving the model, it prints a message indicating that the quantization process has finished.

    Args:
        model (str): The name of the model to be quantized.
        full_model_path (str): The path to the full version of the model.
        out_path (str): The path where the converted model will be saved.
    """
    out_file = f'{out_path}{model}-f32.gguf'

    if os.path.exists(out_file):
        return

    in_string = f"python llama.cpp/convert_hf_to_gguf.py {full_model_path} --outtype f32 --outfile {out_file}"
    print(in_string)
    process = subprocess.Popen(in_string, shell=True, stdout=subprocess.PIPE)
    # wait for the process to finish
    output, error = process.communicate()
    print(f'Finished quantizing {model} to {out_file}')

def quantize_to_type_from_f16(model, model_bf16_path, quant_type, use_f32: bool = False):
    """
    This function takes as input a model name, the path to its f16 version, and a target quantization type.
    It uses the llama-quantize script from llama.cpp to convert the given f16 model to the specified quantization type.
    The converted model is saved in the same path as the f16 model with a filename that includes the model name and the desired quantization type suffix.
    After saving the model, it prints a message indicating that the quantization process has finished.

    Args:
        model (str): The name of the model to be further quantized.
        model_f16_path (str): The path where the f16 version of the model is located.
        quant_type (str): The target quantization type for the model.
    """
    if use_f32:
        m_path = f'{model_bf16_path}{model}-f32.gguf'
    else:
        m_path = f'{model_bf16_path}{model}-bf16.gguf'
    out_file = f'{model_bf16_path}{model}-{quant_type}.gguf'

    in_string = f"llama.cpp/llama-quantize {m_path} {out_file} {quant_type}"
    print('\n' + in_string + '\n')
    process = subprocess.Popen(in_string, shell=True, stdout=subprocess.PIPE)
    # wait for the process to finish
    output, error = process.communicate()
    print(f'Finished quantizing {model}-bf16 to {out_file}')


#def quantize_full_to_types(model: str, full_model_path: str, types: list):
def quantize_full_to_types(model_settings: dict, types: list):
    """
    This function takes as input a model name, the path to its full version, and a list of desired quantization types.
    It first creates a directory to store the quantized versions of the model.
    Then, it uses the `quantize_to_f16` function to convert the given model from Hugging Face format to f16 gguf format.
    After that, for each desired quantization type in the list, it uses the `quantize_to_type_from_f16` function to further quantize the f16 model to the specified quantization type.
    The converted models are saved in the created directory with filenames that include the model name and the desired quantization type suffix.
    After saving all the models, it prints a message indicating that the quantization process has finished for each model.

    Args:
        model (str): The name of the model to be quantized.
        full_model_path (str): The path to the full version of the model.
        types (list): A list of desired quantization types for the model.
    """
    full_model_path = model_settings['full_model_path']
    model = model_settings['model_name']
    use_f32 = model_settings['use_f32']
    quants_path = f'./LLM/quants/{model}/' #f'{full_model_path}/{model}-quants/'

    if not os.path.exists(quants_path):
        os.makedirs(quants_path)
    
    #if 'gemma2b' not in model:
    quantize_to_f16(model=model, full_model_path=full_model_path, out_path=quants_path)
    #quantize_to_f32(model=model, full_model_path=full_model_path, out_path=quants_path)

    
    for q_type in types:
        if not os.path.exists(f'{quants_path}{model}-{q_type}.gguf'):
            quantize_to_type_from_f16(model=model, model_bf16_path=quants_path, quant_type=q_type, use_f32=use_f32)




"""
def main():

    model_f16_path = './LLM'
    model = 'llama3'
    full_model_path = './LLM/Meta-Llama-3-8B-Instruct'
    quantize_full_to_types(model=model, full_model_path=full_model_path, types=['Q4_K_M'])
    print('done')

if __name__ == "__main__":
    main()
"""
