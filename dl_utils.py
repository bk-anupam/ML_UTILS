import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import wandb
from pytorch_lightning.loggers import WandbLogger
import wandb
from kaggle_secrets import UserSecretsClient


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

def wandb_login():
    user_secrets = UserSecretsClient()
    wandb_secret = user_secrets.get_secret("wandb")
    wandb.login(key=wandb_secret)

def get_wandb_logger(fold, config_dict=None):
    logger = None
    wandb_key = None
    if config_dict["RUNTIME"] == "KAGGLE":
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("wandb")
    else:
        wandb_key = config_dict["WANDB_KEY"]    
    wandb.login(key=wandb_key)        
    logger = WandbLogger(
        name=config_dict["WANDB_RUN_NAME"] + f"_fold{fold}", 
        project=config_dict["WANDB_PROJECT"],
        config=config_dict,
        group=config_dict["MODEL_TO_USE"]
    )
    return logger

class MetricsAggCallback(Callback):
    def __init__(self, metric_to_monitor, mode):
        self.metric_to_monitor = metric_to_monitor
        self.metrics = []
        self.best_metric = None
        self.mode = mode
        self.best_metric_epoch = None
        self.val_epoch_num = 0
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.val_epoch_num += 1
        metric_value = trainer.callback_metrics[self.metric_to_monitor].detach().cpu().item()
        val_loss = trainer.callback_metrics["val_loss"].cpu().detach().item()
        current_lr = trainer.callback_metrics["cur_lr"].cpu().detach().item()
        print(f"epoch = {self.val_epoch_num} => metric {self.metric_to_monitor} = {metric_value}, " \
              f"val_loss={val_loss}, lr={current_lr}")
        self.metrics.append(metric_value)
        if self.mode == "max":
            self.best_metric = max(self.metrics)            
        else:
            self.best_metric = min(self.metrics)
        self.best_metric_epoch = self.metrics.index(self.best_metric)
                        