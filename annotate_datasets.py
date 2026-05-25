#!/usr/bin/env python3
"""
Estimate appraisals for full dialogues (prompt + response concatenated) or individual prompts/responses.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure appraise_plm can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from appraise_plm.model import AppraiseATN, APPRAISAL_DIMENSIONS

def load_beavertails(data_dir, split='330k_test'):
    """Load BeaverTails from HuggingFace dataset cache or directly."""
    from datasets import load_dataset
    print(f"Loading BeaverTails split={split} from HuggingFace ...")
    dataset = load_dataset('PKU-Alignment/BeaverTails', split=split)
    df = dataset.to_pandas()
    def _active_categories(cat_dict):
        return ', '.join(k for k, v in cat_dict.items() if v)
    df['category_labels'] = df['category'].apply(_active_categories)
    df = df.drop(columns=['category'])
    return df, 'prompt', 'response'

def load_cosafe(data_dir, use_single_prompt=False):
    """Load CoSafe from JSON files."""
    rows = []
    if use_single_prompt:
        pattern = os.path.join(data_dir, 'Single Prompt', '*.json')
        files = sorted(glob.glob(pattern))
        for fpath in files:
            category = os.path.basename(fpath).replace('_select_100.json', '')
            with open(fpath) as f:
                for line in f:
                    item = json.loads(line.strip())
                    rows.append({'prompt': str(item[0]), 'response': '', 'context': '', 'category': category})
    else:
        pattern = os.path.join(data_dir, 'CoSafe datasets', '*.json')
        files = sorted(glob.glob(pattern))
        for fpath in files:
            category = os.path.basename(fpath).replace('.json', '')
            with open(fpath) as f:
                for line in f:
                    conv = json.loads(line.strip())
                    if not conv: continue
                    last_user_idx = None
                    for i in range(len(conv) - 1, -1, -1):
                        if conv[i]['role'] == 'user':
                            last_user_idx = i
                            break
                    if last_user_idx is None: continue
                    prompt = conv[last_user_idx]['content']
                    ctx_parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in conv[:last_user_idx]]
                    context = ' '.join(ctx_parts)
                    response = conv[last_user_idx - 1]['content'] if last_user_idx > 0 and conv[last_user_idx - 1]['role'] == 'assistant' else ''
                    rows.append({'prompt': prompt, 'response': response, 'context': context, 'category': category})
    return pd.DataFrame(rows), 'prompt', 'response'

def load_prosocial(data_dir, split='all'):
    """Load ProsocialDialog dataset."""
    splits = ['train', 'validation', 'test'] if split == 'all' else [split]
    frames = []
    for s in splits:
        fpath = os.path.join(data_dir, f'prosocial_dialog_{s}_appraised.csv')
        if os.path.exists(fpath):
            df_s = pd.read_csv(fpath, low_memory=False)
            df_s['split'] = s
            frames.append(df_s)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, 'context', 'response'

def load_diasafety(data_dir, split='all'):
    """Load DiaSafety dataset."""
    splits = ['train', 'val', 'test'] if split == 'all' else [split]
    frames = []
    for s in splits:
        fpath = os.path.join(data_dir, f'{s}.json')
        if os.path.exists(fpath):
            with open(fpath) as f:
                df_s = pd.DataFrame(json.load(f))
                df_s['split'] = s
                frames.append(df_s)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if 'context' in df.columns:
        df = df.rename(columns={'context': 'prompt'})
    return df, 'prompt', 'response'

def build_dialogue_texts(df, prompt_col, response_col, sep=' '):
    prompts = df[prompt_col].fillna('').astype(str)
    responses = df[response_col].fillna('').astype(str)
    return [f"{p}{sep}{r}" if r.strip() else p for p, r in zip(prompts, responses)]

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()}

@torch.no_grad()
def estimate_appraisals(model, dataloader, device):
    model.eval()
    all_preds = []
    for batch in tqdm(dataloader, desc="Estimating appraisals"):
        out = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
        all_preds.append(out['predictions'].cpu())
    return torch.cat(all_preds, dim=0).numpy()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.corpus == 'beavertails':
        df, prompt_col, response_col = load_beavertails(args.data_dir, split=args.split or '330k_test')
    elif args.corpus == 'cosafe':
        df, prompt_col, response_col = load_cosafe(args.data_dir, use_single_prompt=False)
    elif args.corpus == 'diasafety':
        df, prompt_col, response_col = load_diasafety(args.data_dir, split=args.split or 'all')
    elif args.corpus == 'prosocial':
        df, prompt_col, response_col = load_prosocial(args.data_dir, split=args.split or 'all')
    else:
        raise ValueError(f"Unknown corpus: {args.corpus}")

    if df.empty:
        print("Error: No data loaded.")
        return

    texts = build_dialogue_texts(df, prompt_col, response_col, sep=args.sep) if args.mode == 'dialogue' else df[prompt_col].fillna('').tolist()

    print(f"\nLoading Appraise-ATN model from: {args.model_dir}")
    model = AppraiseATN.from_pretrained(args.model_dir, device=str(device))
    tokenizer = model.get_tokenizer()

    dataset = TextDataset(texts, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    appraisals = estimate_appraisals(model, dataloader, device)

    prefix = "dialogue_" if args.mode == 'dialogue' else ("prompt_" if args.mode == 'prompt' else "response_")
    for i, dim in enumerate(APPRAISAL_DIMENSIONS):
        df[f'{prefix}{dim}'] = appraisals[:, i]

    out_path = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.endswith('.csv'):
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)
    print(f"✓ Saved {len(df)} annotated samples → {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, choices=['beavertails', 'cosafe', 'diasafety', 'prosocial'])
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['dialogue', 'prompt', 'response'], default='dialogue')
    parser.add_argument('--model_dir', type=str, default='appraise_plm/models/ccc_base_v1')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--sep', type=str, default=' ')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    main(parser.parse_args())
