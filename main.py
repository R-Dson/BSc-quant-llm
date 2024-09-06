import Library.run_eval as run_eval, Library.quantize as quantize, Library.perplexity as perplexity
import os 

"""
    { "Q4_0",   LLAMA_FTYPE_MOSTLY_Q4_0,   " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_1",   LLAMA_FTYPE_MOSTLY_Q4_1,   " 4.78G, +0.4511 ppl @ Llama-3-8B",  },
    { "Q5_0",   LLAMA_FTYPE_MOSTLY_Q5_0,   " 5.21G, +0.1316 ppl @ Llama-3-8B",  },
    { "Q5_1",   LLAMA_FTYPE_MOSTLY_Q5_1,   " 5.65G, +0.1062 ppl @ Llama-3-8B",  },
    { "IQ2_XXS",LLAMA_FTYPE_MOSTLY_IQ2_XXS," 2.06 bpw quantization",            },
    { "IQ2_XS", LLAMA_FTYPE_MOSTLY_IQ2_XS, " 2.31 bpw quantization",            },
    { "IQ2_S",  LLAMA_FTYPE_MOSTLY_IQ2_S,  " 2.5  bpw quantization",            },
    { "IQ2_M",  LLAMA_FTYPE_MOSTLY_IQ2_M,  " 2.7  bpw quantization",            },
    { "IQ1_S",  LLAMA_FTYPE_MOSTLY_IQ1_S,  " 1.56 bpw quantization",            },
    { "IQ1_M",  LLAMA_FTYPE_MOSTLY_IQ1_M,  " 1.75 bpw quantization",            },
    { "Q2_K",   LLAMA_FTYPE_MOSTLY_Q2_K,   " 2.96G, +3.5199 ppl @ Llama-3-8B",  },
    { "Q2_K_S", LLAMA_FTYPE_MOSTLY_Q2_K_S, " 2.96G, +3.1836 ppl @ Llama-3-8B",  },
    { "IQ3_XXS",LLAMA_FTYPE_MOSTLY_IQ3_XXS," 3.06 bpw quantization",            },
    { "IQ3_S",  LLAMA_FTYPE_MOSTLY_IQ3_S,  " 3.44 bpw quantization",            },
    { "IQ3_M",  LLAMA_FTYPE_MOSTLY_IQ3_M,  " 3.66 bpw quantization mix",        },
    { "Q3_K",   LLAMA_FTYPE_MOSTLY_Q3_K_M, "alias for Q3_K_M"                   },
    { "IQ3_XS", LLAMA_FTYPE_MOSTLY_IQ3_XS, " 3.3 bpw quantization",             },
    { "Q3_K_S", LLAMA_FTYPE_MOSTLY_Q3_K_S, " 3.41G, +1.6321 ppl @ Llama-3-8B",  },
    { "Q3_K_M", LLAMA_FTYPE_MOSTLY_Q3_K_M, " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
    { "Q3_K_L", LLAMA_FTYPE_MOSTLY_Q3_K_L, " 4.03G, +0.5562 ppl @ Llama-3-8B",  },
    { "IQ4_NL", LLAMA_FTYPE_MOSTLY_IQ4_NL, " 4.50 bpw non-linear quantization", },
    { "IQ4_XS", LLAMA_FTYPE_MOSTLY_IQ4_XS, " 4.25 bpw non-linear quantization", },
    { "Q4_K",   LLAMA_FTYPE_MOSTLY_Q4_K_M, "alias for Q4_K_M",                  },
    { "Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S, " 4.37G, +0.2689 ppl @ Llama-3-8B",  },
    { "Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M, " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    { "Q5_K",   LLAMA_FTYPE_MOSTLY_Q5_K_M, "alias for Q5_K_M",                  },
    { "Q5_K_S", LLAMA_FTYPE_MOSTLY_Q5_K_S, " 5.21G, +0.1049 ppl @ Llama-3-8B",  },
    { "Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M, " 5.33G, +0.0569 ppl @ Llama-3-8B",  },
    { "Q6_K",   LLAMA_FTYPE_MOSTLY_Q6_K,   " 6.14G, +0.0217 ppl @ Llama-3-8B",  },
    { "Q8_0",   LLAMA_FTYPE_MOSTLY_Q8_0,   " 7.96G, +0.0026 ppl @ Llama-3-8B",  },
    { "F16",    LLAMA_FTYPE_MOSTLY_F16,    "14.00G, +0.0020 ppl @ Mistral-7B",  },
    { "BF16",   LLAMA_FTYPE_MOSTLY_BF16,   "14.00G, -0.0050 ppl @ Mistral-7B",  },
    { "F32",    LLAMA_FTYPE_ALL_F32,	   "26.00G              @ 7B",          },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",   LLAMA_FTYPE_ALL_F32,	   "only copy tensors, no quantizing",  },

"""

# Dictionary containing settings for each model
MODEL_SETTINGS = {
    # Settings for the gemma-2b model
    'gemma-2b': {
        'full_model_path' : './LLM/gemma-1.1-2b-it/',  # Full path to the original model
        'model_path' : './LLM/quants/gemma2b-v1.1/',   # Path where quantized models will be saved
        'model_name' : 'gemma2b-v1.1',                 # Name for the quantized model
        'use_f32': False,                              # Whether to use full precision (float32) or not
        'k-quants' : ['Q8_0', 'Q6_K', 'Q5_K', 'Q4_K', 'Q3_K', 'Q2_K']  # List of quantization types to be used
    },
    # Similar settings for other models...
    'phi-3-mini': {
        'full_model_path' : './LLM/Phi-3-mini-4k-instruct/',
        'model_path' : './LLM/quants/phi-3-mini-3b/',
        'model_name' : 'phi-3-mini-3b',
        'use_f32': False,
        'k-quants' : [ 'Q8_0', 'Q6_K', 'Q5_K', 'Q4_K', 'Q3_K', 'Q2_K' ] # 'bf16',
        # 'context': 4096,
        # 'max_tokens': 2048
    },
    'llama3-8b': {
        'full_model_path' : './LLM/Meta-Llama-3-8B-Instruct/',
        'model_path' : './LLM/quants/llama3-8b/',
        'model_name' : 'llama3-8b',
        'use_f32': False,
        'k-quants' : ['Q8_0', 'Q6_K', 'Q4_K', 'Q3_K', 'Q2_K']
        # 'context': 4096,
        # 'max_tokens': 2048

    },
    'gemma-2-9b': {
        'full_model_path' :'./LLM/gemma-2-9b-it/',
        'model_path' : './LLM/quants/gemma-2-9b/',
        'model_name' : 'gemma-2-9b',
        'use_f32': False,
        'k-quants' : ['Q8_0', 'Q6_K', 'Q4_K', 'Q3_K', 'Q2_K']
        # 'context': 4096,
        # 'max_tokens': 2048
    },
    'llama3-70b': {
        'full_model_path' : './LLM/Meta-Llama-3-70B-Instruct/',
        'model_path' : './LLM/quants/llama3-70b/',
        'model_name' : 'llama3-70b',
        'use_f32': False,
        'k-quants' : ['Q8_0', 'Q4_K', 'Q2_K']

    },
    'phi-3-medium': {
        'full_model_path' : './LLM/Phi-3-medium-4k-instruct/',
        'model_path' : './LLM/quants/phi-3-medium-14b/',
        'model_name' : 'phi-3-medium-14b',
        'use_f32': False,
        'k-quants' : [ 'Q8_0', 'Q5_K', 'Q3_K', 'Q2_K' ] # 'bf16',
        
    },
    'gemma-2-27b': {
        'full_model_path' : './LLM/gemma-2-27b-it/',
        'model_path' : './LLM/quants/gemma-2-27b/',
        'model_name' : 'gemma-2-27b',
        'use_f32': False,
        'k-quants' : ['Q8_0', 'Q5_K', 'Q3_K', 'Q2_K']
    }
}
"""
'phi-3-small': {
        'full_model_path' : './LLM/Phi-3-small-8k-instruct/',
        'model_path' : './LLM/quants/phi-3-small-7b/',
        'model_name' : 'phi-3-small-7b',
        'use_f32': False,
        'k-quants' : [ 'bf16' ]#[ 'Q8_0', 'Q6_K', 'Q4_K', 'Q3_K', 'Q2_K' ] # ,
        
    }"""
# Dictionary containing settings for each dataset
DATASET_SETTINGS = {
    # Settings for the TIGER-Lab/MMLU-Pro dataset
    'TIGER-Lab/MMLU-Pro':{
        'n_shot': 1,                                   # Number of shots to use for few-shot learning
        'temperature': 0.1,                            # Temperature parameter for sampling from the model
        'context': 4096,                               # Maximum context length
        'max_tokens': 2048                             # Maximum number of tokens to generate
    },
    # Similar settings for other datasets...
    # We use 10 handwritten few-shot examples, 5 using str functions and 5 using list functions. For each prompt, we include two few-shot examples, one string few-shot example and one list few-shot example, for a total of 25 different combinations of few-shot prompts. We generate programs and inputs using Code Llama 34B with temperature T=1.
    'cruxeval-org/cruxeval': {
        'n_shot': 1,
        'temperature': 0.2,
        'context': 4096,
        'max_tokens': 2048,
        'mode': 'O' # 'I' # https://crux-eval.github.io/leaderboard.html
    },
    'TAUR-Lab/MuSR': { # 5.2 BENCHMARKING WITH MUSR https://arxiv.org/pdf/2310.16049
        'n_shot': 0,
        'temperature': 0.1,
        'context': 4096,
        'max_tokens': 2048
    }
}
# Set the model and dataset to be used in this run
MODEL_TO_RUN = 'llama3-8b'
MODEL_TO_RUN = 'gemma-2b'
MODEL_TO_RUN = 'phi-3-medium'
MODEL_TO_RUN = 'gemma-2-9b'
MODEL_TO_RUN = 'phi-3-mini'
MODEL_TO_RUN = 'gemma-2-27b'

MODEL_TO_RUN = 'llama3-70b'



DATASETS = [ 'TIGER-Lab/MMLU-Pro', 'TAUR-Lab/MuSR', 'cruxeval-org/cruxeval' ]

# https://github.com/ggerganov/llama.cpp/pull/7992/files

def main():
    # Loop over each model in MODEL_SETTINGS
    for model in MODEL_SETTINGS:
        # Skip models that don't contain '27' in their name
        if '27' not in model:
            continue
        MODEL_TO_RUN = model
        # run quantize to generate the quants
        settings = MODEL_SETTINGS[MODEL_TO_RUN]
        path = settings['model_path']
        name = settings['model_name']

        K_QUANTS = settings['k-quants']
        K_QUANTS.reverse() # Reverse the list of quantization types to start with the smallest ones
        
        # Quantize the model for each quantization type in K_QUANTS
        quantize.quantize_full_to_types(model_settings=settings, types=K_QUANTS)

        PPL = [] # List to store perplexity results for this model
        # Loop over each quantized model and evaluate it on the datasets
        for quant_type in K_QUANTS:

            model_name = name + f'-{quant_type}'
            print(f'Evaluating: {model_name}')
            model_q_path = f'{path}{model_name}.gguf'

            # Calculate and store the perplexity of the quantized model
            try:
                if not os.path.exists(f'results/{name}/{model_name}-PPL.json'):
                    estimate, margin_of_error = perplexity.perplexity(model_q_path)
                    PPL.append({quant_type: {'estimate':estimate, 'margin_of_error':margin_of_error}})
                    perplexity.save_perplexity(name, PPL, quant_type)
            except:
                pass

            abs_path = os.path.abspath(model_q_path)
            # Evaluate the quantized model on each dataset and store the results
            for dataset in DATASETS:
                data_remove = dataset.replace('/', '-')
                name_path = f'results/{model_name}/{model_name}-{data_remove}-results.json'
                # Only evaluate the model on this dataset if results don't already exist
                if not os.path.exists(name_path) and os.path.exists(abs_path):
                    run_eval.evaluate(model_name=model_name, dataset_name=dataset, dataset_settings=DATASET_SETTINGS[dataset], model_path=abs_path)
                    pass
            
        # perplexity
        # perplexity.save_perplexity(name, PPL)
    
if __name__ == "__main__":
    main()
