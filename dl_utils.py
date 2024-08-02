import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
from transformers import get_linear_schedule_with_warmup

def get_linear_lr_scheduler(optimizer, config_dict):
    # Scheduler and math around the number of training steps.    
    num_train_steps = config_dict["NUM_EPOCHS"] * config_dict["STEPS_PER_EPOCH"]
    num_warmup_steps = int(config_dict["MODEL_PARAMS"]["warmup_prop"] * config_dict["NUM_EPOCHS"] * config_dict["STEPS_PER_EPOCH"])    
    print(f"num_train_steps = {num_train_steps}")
    print(f"num_warmup_steps = {num_warmup_steps}")
    lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
    return lr_scheduler    

def get_optimizer(lr, params, config_dict):
    model_optimizer = None
    interval = "epoch"
    if config_dict["SCHEDULER"] != "ReduceLROnPlateau":
        model_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, params), 
                lr=lr,
                weight_decay=config_dict["WEIGHT_DECAY"]
            )        
    if config_dict["SCHEDULER"] == "CosineAnnealingWarmRestarts":
        lr_scheduler = CosineAnnealingWarmRestarts(
                            model_optimizer, 
                            T_0=config_dict["T_0"], 
                            T_mult=1, 
                            eta_min=config_dict["MIN_LR"], 
                            last_epoch=-1
                        )
    elif config_dict["SCHEDULER"] == "OneCycleLR":
        lr_scheduler = OneCycleLR(
            optimizer=model_optimizer,
            max_lr=config_dict["MAX_LR"],
            epochs=config_dict["NUM_EPOCHS"],
            steps_per_epoch=config_dict["STEPS_PER_EPOCH"],
            verbose=True
        )
        interval = "step"
    elif config_dict["SCHEDULER"] == "CosineAnnealingLR":
        lr_scheduler = CosineAnnealingLR(model_optimizer, eta_min=config_dict["MIN_LR"], T_max=config_dict["NUM_EPOCHS"])
    elif config_dict["SCHEDULER"] == "LinearWithWarmup":
        lr_scheduler = get_linear_lr_scheduler(model_optimizer)
        interval = "step"
    else:
        # ReduceLROnPlateau throws an error is parameters are filtered, 
        # refer: https://github.com/PyTorchLightning/pytorch-lightning/issues/8720
        model_optimizer = torch.optim.Adam(
            params, 
            lr=lr,
            weight_decay=config_dict["WEIGHT_DECAY"]
        )
        print(f"param groups count = {len(model_optimizer.param_groups)}")  
        lr_scheduler = ReduceLROnPlateau(
                            model_optimizer, 
                            mode="min",                                                                
                            factor=0.1,
                            patience=config_dict["SCHEDULER_PATIENCE"],
                            min_lr=config_dict["MIN_LR"],
                            verbose=True
                        )   
    return {
        "optimizer": model_optimizer, 
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": interval,
            "monitor": "val_loss",
            "frequency": 1
        }
    }