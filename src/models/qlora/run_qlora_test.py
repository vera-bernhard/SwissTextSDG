import os
import sys
import copy

sys.path.append(os.getcwd())
from src.models.qlora.qlora_model import QloraModel
from src.models.mbert.pytorch_model import PyTorchModel
from src.data.dataset import PytorchDataset, SwissTextDataset
from src.data.tokenizer import SwissTextTokenizer
from torch.utils.data import Dataset, DataLoader

from src.helpers.seed_helper import initialize_gpu_seed
from src.models.mbert.config import read_arguments_test, update_args_with_config
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *

import torch
from transformers import (AutoTokenizer,
                          MistralForSequenceClassification, 
                          BitsAndBytesConfig, 
                          Trainer, 
                          TrainingArguments)
from peft import (LoraConfig, 
                  PeftConfig, 
                  PeftModel, 
                  get_peft_model,
                  prepare_model_for_kbit_training)
from sklearn.metrics import f1_score


setup_logging()

def high_level_test(adapter_checkpoint):
    bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= 'nf4',
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= True,
)
    
    model_checkpoint = 'mistralai/Mistral-7B-v0.1'
    model = MistralForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=18,
            quantization_config=bnb_config,
            device_map='auto',
    )
    
    m = PeftModel.from_pretrained(model, adapter_checkpoint)
    m = m.merge_and_unload()
    tokenizer  = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    text = "This is an abstract about water filtering"
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)

    return outputs

def main(args):
    initialize_gpu_seed(args.model_seed)

    checkpoint_suffix = '__epoch' + str(args.epoch)
    if args.epoch == 0:
        checkpoint_suffix += '__zeroshot'

    file_name = "".join([args.model_name, checkpoint_suffix])
    checkpoint_path = experiment_file_path(args.experiment_name, file_name)

    model = QloraModel.load_from_checkpoint(args, checkpoint_path)
    
    if args.other_testset:
        test_data_loader = SwissTextDataset.create_test_instance(args)
        model.test_data_loader = test_data_loader
        model.test(args.epoch)
        
    # outputs = high_level_test(checkpoint_path)

if __name__ == '__main__':
    args = read_arguments_test()
    main(args)
