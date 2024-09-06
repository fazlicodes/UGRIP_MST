from datasets import load_dataset, get_dataset_config_names
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, LlamaForCausalLM,AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import evaluate
import argparse 
from tqdm import tqdm
import time
import scipy.stats
import os
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

# Log in to Hugging Face with your API key
login(token="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")  


SEED = 42
NUM_PROC=5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
CACHE=None

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def dataloader_evaluate(dataset_name, label, model_name):
    lang_accuracies = {}
    clf_metrics = evaluate.load("accuracy")

    langs = ['default']
    for lang_code in langs:
        print(f"Processing {lang_code}...")
        predictions = []
        actual_labels = []

        dataset = load_dataset(f'pkavumba/{dataset_name}', lang_code, split='test', cache_dir=CACHE)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        inf_time_stats = []

        for i, batch in tqdm(enumerate(dataloader), total=100):
            if i == 100:
                break
            inf_time_for_batch = 0.0
            premise_field = batch['premise'][0]
            question_field = batch['question'][0]
            choice1_field = batch['choice1'][0]
            choice2_field = batch['choice2'][0]
            prompt_text = f"The {question_field} of {premise_field} is that 1. {choice1_field} or 2. {choice2_field}. Choose the correct option."
            inputs = tokenizer(prompt_text, return_tensors='pt', padding=False, truncation=True).to(device)

            start_time = time.time()
            outputs = model(**inputs)
            end_time = time.time()

            inf_time_for_batch = (end_time - start_time)
            inf_time_stats.append(inf_time_for_batch)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch[label]
            predictions.extend(preds.tolist())
            actual_labels.extend(labels)

        # lang_accuracies[lang_code] = f"{clf_metrics.compute(predictions=predictions, references=actual_labels)['accuracy']*100:.2f}"

        mean, ci = mean_confidence_interval(inf_time_stats)
        print(f"Mean Inference Time: {mean:.4f} s, CI: {ci:.4f}")

    df = pd.DataFrame(inf_time_stats, columns=['Inference Time'])
    
    return mean, ci, df



def initialize_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=True)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, use_auth_token=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_ids = ["Qwen/Qwen2-0.5B-Chat", "Qwen/Qwen2-1.5B-Chat", "Qwen/Qwen2-72B-Chat", 'CohereForAI/aya-23-8B', 'CohereForAI/aya-23-35B',
                'google/gemma-2-2b', 'google/gemma-2-9b', 'google/gemma-2-27b','meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-70B']
    
    model_ids = ["Qwen/Qwen2-0.5B-Chat", "Qwen/Qwen2-1.5B-Chat", 'CohereForAI/aya-23-8B', 'google/gemma-2-2b', 'google/gemma-2-9b','meta-llama/Meta-Llama-3.1-8B']
    df = pd.DataFrame(columns=['Model', 'Inference Time Mean', 'Inference Time CI'])
    results = []

    if not os.path.exists('ind_llm_results'):
        os.makedirs('ind_llm_results')

    for model_id in tqdm(model_ids):
        print(f"Model: {model_id}")
        model, tokenizer = initialize_model_and_tokenizer(model_id, model_id, device)
        model_name = model_id.split('/')[1]
        mean, ci, df = dataloader_evaluate('balanced-copa', 'label', model_name)
        df.to_csv(f'ind_llm_results/{model_name}_balanced-copa_inference_time.csv', index=False)
        results.append({'Model': model_id, 'Inference Time Mean': mean, 'Inference Time CI': ci})

    df = pd.DataFrame(results)
    df.to_csv('llm_inference_time_results.csv', index=False)