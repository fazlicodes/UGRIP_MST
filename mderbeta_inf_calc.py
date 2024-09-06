from datasets import load_dataset, get_dataset_config_names
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import evaluate
import argparse
from tqdm import tqdm
import time
import scipy.stats
from huggingface_hub import login

# Log in to Hugging Face with your API key
login(token="api_key") 

SEED = 42
NUM_PROC = 5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
CACHE = None

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

    
def dataloader_evaluate(dataset_name, label, batch_size):
    lang_accuracies = {}
    langs = get_dataset_config_names(f"mbzuai-ugrip-statement-tuning/{dataset_name}")
    clf_metrics = evaluate.load("accuracy")
    
    for lang_code in langs:
        print(f"Processing {lang_code}...")
        predictions = []
        actual_labels = []
        inf_time_stats = []

        dataset = load_dataset(f'mbzuai-ugrip-statement-tuning/{dataset_name}', lang_code, split='test', cache_dir=CACHE)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        statement_fields = dataset.column_names
        statement_fields.remove(label)

        for batch in tqdm(dataloader):
            probabilities = []
            batch_inference_time = 0.0 

            for statement_field in statement_fields:
                tok = tokenizer(batch[statement_field], return_tensors='pt', padding=True).to(device)

                start_time = time.time()
                logits = model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).logits
                end_time = time.time()

                probabilities.append(F.softmax(logits, dim=-1)[:, 1])
                batch_inference_time += (end_time - start_time)

            inf_time_stats.append(batch_inference_time)

            labels = batch[label]
            preds = torch.argmax(torch.stack(probabilities, dim=-1), dim=-1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

        lang_accuracies[lang_code] = f"{clf_metrics.compute(predictions=predictions, references=actual_labels)['accuracy'] * 100:.2f}"

        mean, ci = mean_confidence_interval(inf_time_stats)
        print(f"Mean Inference Time: {mean:.4f} s, CI: {ci:.4f}")

    return mean, ci


def initialize_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, torch_dtype=torch.bfloat16, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, use_auth_token=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help='HuggingFace path to model', default='microsoft/mdeberta-v3-base')
    parser.add_argument("--tokenizer", help="HuggingFace tokenizer type", default='microsoft/mdeberta-v3-base')
    parser.add_argument("--cache", help='cachedir for HuggingFace datasets', default=None)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = initialize_model_and_tokenizer(args.model, args.tokenizer, device)
    df = pd.DataFrame(columns=['Nos Labels', 'Inference Time Mean', 'Inference Time CI'])
    results = []
    
    nos_labels = [2, 4, 8, 16, 32]
    for batch_size in tqdm(nos_labels):
        print(f"Zero-shot classification for {batch_size} labels")
        mean, ci = dataloader_evaluate('copa', 'label', batch_size)
        results.append({'Nos Labels': batch_size, 'Inference Time Mean': mean, 'Inference Time CI': ci})

    df = pd.DataFrame(results)
    df.to_csv('mderbeta_inference_time_results.csv', index=False)
