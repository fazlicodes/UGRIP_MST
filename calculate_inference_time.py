import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm import tqdm
import time
import scipy.stats
import os
from huggingface_hub import login
import warnings

warnings.filterwarnings("ignore")

# Log in to Hugging Face with your API key
login(token="api_key")  

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def simulate_evaluation(model, tokenizer, device, batch_size):
    inf_time_stats = []
    labels = ['label 1', 'label 2']
    
    for _ in tqdm(range(128 // batch_size)):  # Adjust number of iterations to simulate different batch sizes
        batch_texts = [f"This is a simulation of zero-shot classification task. Your task, given this text: 'This is dummy text' is to assign which of the following labels: {labels}, suited to that text?" for _ in range(batch_size)]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)

        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()

        inf_time_for_batch = (end_time - start_time)
        inf_time_stats.append(inf_time_for_batch)

    mean, ci = mean_confidence_interval(inf_time_stats)
    print(f"Mean Inference Time: {mean:.4f} s, CI: {ci:.4f}")

    df = pd.DataFrame(inf_time_stats, columns=['Inference Time'])
    return mean, ci, df

def initialize_model_and_tokenizer(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    if 'mdeberta' in model_id:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_auth_token=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer

if __name__ == "__main__":  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    model_id ='microsoft/mdeberta-v3-base'

    if not os.path.exists('ind_results'):
        os.makedirs('ind_results')

    batch_sizes = [2, 4, 8, 16, 32]

    print(f"Model: {model_id}")
    model, tokenizer = initialize_model_and_tokenizer(model_id, device)
    model_name = model_id.split('/')[1]
    
    for batch_size in batch_sizes:
        print(f"Evaluating with batch size: {batch_size}")
        mean, ci, df = simulate_evaluation(model, tokenizer, device, batch_size)
        df.to_csv(f'ind_results/{model_name}_batch_{batch_size}_inference_time.csv', index=False)
        results.append({'Model': model_id, 'Batch Size': batch_size, 'Inference Time Mean': mean, 'Inference Time CI': ci})

    df = pd.DataFrame(results)
    df.to_csv('mderbeta_inference_time_results.csv', index=False)

    model_ids = ["Qwen/Qwen2-0.5B-Chat", "Qwen/Qwen2-1.5B-Chat", "Qwen/Qwen2-72B-Chat", 'CohereForAI/aya-23-8B', 'CohereForAI/aya-23-35B',
            'google/gemma-2-2b', 'google/gemma-2-9b', 'google/gemma-2-27b','meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-70B']
    llm_results = []

    for model_id in tqdm(model_ids):
        print(f"Model: {model_id}")
        model, tokenizer = initialize_model_and_tokenizer(model_id, device)
        model_name = model_id.split('/')[1]
        
        mean, ci, df = simulate_evaluation(model, tokenizer, device, batch_size=1)
        df.to_csv(f'ind_results/{model_name}_inference_time.csv', index=False)
        llm_results.append({'Model': model_id,'Inference Time Mean': mean, 'Inference Time CI': ci})

    llm_df = pd.DataFrame(llm_results)
    llm_df.to_csv('llm_inference_time_results.csv', index=False)
