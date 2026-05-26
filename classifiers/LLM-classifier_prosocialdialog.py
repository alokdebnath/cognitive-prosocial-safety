from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed, BitsAndBytesConfig
import os
from sklearn.model_selection import train_test_split
import argparse
import logging
from scipy.special import softmax
import bitsandbytes as bnb
from peft import LoraConfig, PeftConfig, PeftModel, AutoPeftModelForCausalLM, TaskType, AutoPeftModelForSequenceClassification, prepare_model_for_kbit_training, get_peft_model
import torch
import torch.nn.functional as F
from accelerate import PartialState
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
import re
from confusables import confusable_characters
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

from numba import cuda
import nvidia_smi, psutil

def report_gpu():
  nvidia_smi.nvmlInit()
  handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
  info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
  print("GPU [GB]:", f'{info.used/1024/1024/1024:.2f}', "/", f'{info.total/1024/1024/1024:.1f}')
  nvidia_smi.nvmlShutdown()
  print('RAM [GB]:', f'{psutil.virtual_memory()[3]/1024/1024/1024:.2f}', "/", f'{psutil.virtual_memory()[0]/1024/1024/1024:.1f}')

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

H_PROBABILITY = 0.05
ZWJ_PROBABILITY = 0.05

def homoglyph(input_text):
    output = ""
    for char in input_text:
      if (random.random() < H_PROBABILITY):
        output += confusable_characters(char)[int(random.random()*len(confusable_characters(char)))]
      else:
        output += char
      if (random.random() < ZWJ_PROBABILITY):
        output += "\u200D"
    return output

def preprocess(text):
  EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")                          # e.g., name@example.com
  USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")                                                 # e.g., @my_username
  PHONE_PATTERN = re.compile(r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}") #modified from https://stackabuse.com/python-regular-expressions-validate-phone-numbers/
  text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
  text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
  text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace('  [PHONE]', ' [PHONE]')
  return text.lower().strip()

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, padding=True, max_length=512)

SAFETY_ORDER = {
    '__casual__': 4,
    '__possibly_needs_caution__': 3,
    '__probably_needs_caution__': 2,
    '__needs_caution__': 1,
    '__needs_intervention__': 0,
}

SAFETY_TIER_LABELS = {
    'high_safety': ['__casual__', '__possibly_needs_caution__'],
    'low_safety':  ['__needs_caution__', '__needs_intervention__'],
}

def get_data(train_path, dev_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    
    train_df['text'] = train_df['context']
    train_df['label'] = train_df.safety_label.map(SAFETY_ORDER)
    
    val_df['text'] = val_df['context']
    val_df['label'] = val_df.safety_label.map(SAFETY_ORDER)
    
    test_df['text'] = test_df['context']
    test_df['label'] = test_df.safety_label.map(SAFETY_ORDER)
    
    return train_df, val_df, test_df

f1_metric = evaluate.load("f1")
metric = evaluate.load("bstrai/classification_report")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results

weights = [1.0, 1.0, 1.0, 1.0, 1.0]
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        #loss = F.binary_cross_entropy_with_logits(logits[:,1], labels.to(torch.float32))#, pos_weight=self.label_weights)
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def find_all_linear_names(model):
    lora_module_names = set()
    #print(list(model.named_modules()))
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if "base_layer" in lora_module_names:  # problem with training from peft checkpoint
        lora_module_names.remove("base_layer")
    return list(lora_module_names)

def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model, continue_train):
    global weights
    weights = 1/train_df['label'].value_counts(normalize=True).sort_index().to_numpy()
    print(weights)

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    floatorbfloat = torch.bfloat16
    if 'lama' in model.lower() or 'gemma' in model.lower():
        floatorbfloat = torch.bfloat16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=floatorbfloat,
        bnb_4bit_quant_storage=floatorbfloat,
    )
    
    model_name = model.split("/")[-1].lower()
    
    if ('deberta' in model_name) or ('roberta' in model_name):
      bnb_config=None
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id, trust_remote_code=True, quantization_config=bnb_config, device_map="auto" if 'berta' not in model_name else None, torch_dtype="auto", 
    )
    model.config.use_cache = False
     
    #fix a buf in num_labels from config propagation in qwen3.5
    if model.config.num_labels != len(label2id):
      model.config.num_labels = len(label2id)
      model.score = torch.nn.Linear(model.score.in_features, len(label2id), bias=False)
         
    #DM added
    if tokenizer.pad_token is None:
      if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
      else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    try:
      model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
      print("Warning: Exception occured while setting pad_token_id")
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    
    target_modules=[]
    if 'falcon' in model_name:
      target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif 'mistral' in model_name:
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    #elif 'llama' in model_name:
    #  pass #target_modules=['v_proj', 'q_proj', 'k_proj', 'o_proj'] #'down_proj', 'up_proj', 'gate_proj', #['v_proj', 'up_proj', 'gate_proj', 'o_proj', 'down_proj', 'k_proj', 'q_proj']
    #elif 'gemma' in model_name:
    #  target_modules=["q_proj", "k_proj"]
    elif 'deberta' in model_name:
      target_modules=["query_proj", "key_proj", "value_proj"]
    elif 'bert' in model_name:
      target_modules=["query", "value"]
    else:
      target_modules=find_all_linear_names(model)
    print(target_modules)
    
    modules_to_save=["score"]
    if ('deberta' in model_name) or ('roberta' in model_name): modules_to_save=["classifier", "pooler"]
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules, #"all-linear", #
        modules_to_save=modules_to_save
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    output_dir = checkpoints_path
    if continue_train:
      output_dir = output_dir.split('/checkpoint-')[0]
    
    fp16=False
    bf16=True
    tf32=False
    if ('deberta' in model_name) or ('roberta' in model_name):
      tf32=True
      bf16=False
    # create Trainer 
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16 if '14b' not in model_name else 8,#8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=1000,
        learning_rate=2e-5,
        fp16=fp16,
        bf16=bf16,
        tf32=tf32,
        num_train_epochs=3,
        warmup_ratio=0.03,
        #group_by_length=True,
        #lr_scheduler_type="constant",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": True},
        load_best_model_at_end=True,
        #metric_for_best_model='AUC',
        #evaluation_strategy="steps",
        eval_strategy="steps",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        #tokenizer=tokenizer,
        processing_class=tokenizer, #newest transformers
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if continue_train:
      trainer.train(resume_from_checkpoint=True)
    else:
      trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)
    trainer.model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    for module in modules_to_save:
      try:
        torch.save(getattr(trainer.model,module).state_dict(), f'{best_model_path}/{module}-params.pt')
      except:
        print(f"Module {module} not dumped.")
    
    try:
      report_gpu()
    except:
      pass


def test(test_df, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    peft_config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
       peft_config.base_model_name_or_path, num_labels=len(label2id), id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16 if 'bert' not in model_path else "auto", device_map="auto"#, ignore_mismatched_sizes=True
    )
    
    #fix a buf in num_labels from config propagation in qwen3.5
    if model.config.num_labels != len(label2id):
      model.config.num_labels = len(label2id)
      model.score = torch.nn.Linear(model.score.in_features, len(label2id), bias=False, dtype=torch.bfloat16 if 'bert' not in model_path else "auto")
    
    #DM added
    if tokenizer.pad_token is None:
      if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
      else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    try:
      model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
      print("Warning: Exception occured while setting pad_token_id")
      
    peft_model = PeftModel.from_pretrained(model, model_path)
    model=peft_model
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        #tokenizer=tokenizer,
        processing_class=tokenizer, #newest transformers
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    probs = [x[y] for x,y in zip(prob_pred, preds)]
    results = None
    try:
      results = metric.compute(predictions=preds, references=predictions.label_ids)
    except:
      pass
    # return dictionary of classification report
    
    try:
      report_gpu()
    except:
      pass
    return results, preds, probs, prob_pred


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--dev_file_path", "-d", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
    parser.add_argument("--random_seed", "-rs", required=False, help="Random seed.", type=int, default=0)
    parser.add_argument('--test_only', '-to', action='store_true')
    parser.add_argument('--continue_train', '-c', action='store_true')

    args = parser.parse_args()

    random_seed = args.random_seed
    train_path = args.train_file_path # For example 'train.jsonl'
    dev_path = args.dev_file_path # For example 'dev.jsonl'
    test_path = args.test_file_path # For example 'devtest.jsonl'
    model = args.model # For example 'xlm-roberta-base'
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path # For example predictions.jsonl
    
    label2id = SAFETY_ORDER
    id2label = {}
    for k,v in label2id.items(): id2label[v] = k

    set_seed(random_seed)

    #get data for train/valid/test sets
    train_df, valid_df, test_df = get_data(train_path, dev_path, test_path, random_seed)
    train_df = train_df.dropna(subset=['text'])
    valid_df = valid_df.dropna(subset=['text'])
    print(train_df.label.value_counts())
    print(valid_df.label.value_counts())
    print(test_df.label.value_counts())
    
    try:
      report_gpu()
    except:
      pass
    
    if True:    
      # train detector model
      if args.test_only != True:
        fine_tune(train_df[['text', 'label']], valid_df[['text', 'label']], f"models/{model}_context", id2label, label2id, model, args.continue_train)
      
      torch.cuda.empty_cache()
      try:
        report_gpu()
      except:
        pass
      
      # test detector model
      if 'label' not in test_df.columns: test_df['label'] = 0
      results, predictions, probs, class_prob = test(test_df[['text', 'label']], f"models/{model}_context/best/", id2label, label2id)
      
      logging.info(results)
      if "id" not in test_df.columns:
          test_df['id'] = test_df.index
      predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions, 'probs': probs})
      for i in range(0, class_prob.shape[1]):
        predictions_df[f'{i}_probs'] = class_prob[:,i]
      predictions_df.to_csv(prediction_path, index=False)
      