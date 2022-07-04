import math
import torch
import transformers
import bitsandbytes as bnb
from datasets import Dataset

def get_fold_ds(fold, df, preprocess_data):
    """
    Returns train and validation hugging face datasets corresponding to a fold
    Args:
        fold: fold number
        df: train dataframe
        preprocess_data: partial function with logic to tokenize text data
    """
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    ds_train_raw = Dataset.from_pandas(train_df)
    ds_valid_raw = Dataset.from_pandas(valid_df)
    raw_ds_col_names = ds_train_raw.column_names    
    ds_train = ds_train_raw.map(preprocess_data, batched=True, batch_size=1000, remove_columns=raw_ds_col_names)
    ds_valid = ds_valid_raw.map(preprocess_data, batched=True, batch_size=1000, remove_columns=raw_ds_col_names)    
    return train_df, valid_df, ds_train, ds_valid

# Thanks to Nicolas Broad. Taken from https://www.kaggle.com/code/nbroad/8-bit-adam-optimization/notebook
def get_optimizer(model, args, train_dataset, adam_bits=32):    
    '''
    Creates an 32 bit or 8 bit AdamW optimizer and learning rate scheduler instances. 
    8 bit Adam optimizer is key to training large transformer models on 16GB GPUs.
    Args:
        model: transformer model to train
        args: hugging face training arguments
        train_dataset: hugging face train dataset
        adam_bits:  whether to create 32 bit adam optimizer (default) or 8 bit
    '''
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # These are the only changes you need to make. The first part sets the optimizer to use 8-bits
    # The for loop sets embeddings to use 32-bits
    if adam_bits == 32:
        optimizer = bnb.optim.Adam32bit(optimizer_grouped_parameters, lr=args.learning_rate)
    if adam_bits == 8:
        optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.learning_rate)
        
    # Thank you @gregorlied https://www.kaggle.com/nbroad/8-bit-adam-optimization/comments#1661976
    if adam_bits == 8:
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )            

    num_update_steps_per_epoch = len(train_dataset) // args.per_device_train_batch_size // args.gradient_accumulation_steps
    if args.max_steps == -1 or args.max_steps is None:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        num_train_epochs = args.max_steps / num_update_steps_per_epoch        
        args.num_train_epochs = math.ceil(num_train_epochs)
        
    if args.warmup_ratio is not None:
        args.num_warmup_steps = int(args.warmup_ratio * args.max_steps)

    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_steps,
    )
    return optimizer, lr_scheduler        