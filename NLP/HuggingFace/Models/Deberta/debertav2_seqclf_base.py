import torch
import torch.nn as nn
from loss_functions import CorrLoss, LossType
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout

# base class for implementing custom heads on top of deberta-v2 backbone
class DebertaV2ForSeqClfBase(DebertaV2PreTrainedModel):    
    def __init__(self, config, loss_type: str = None):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.loss_fn = None
        # if a specific loss type has been specified, use it
        if loss_type is not None:
            if loss_type == LossType.MSE:
                self.loss_fn = nn.MSELoss()
            elif loss_type == LossType.BCE_WITH_LOGITS:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
            elif loss_type == LossType.PEARSON:
                self.loss_fn = CorrLoss()
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)
        
    def get_loss(self, logits, labels):
        loss = None
        # custom logic to handle regression problems with explicitly specified loss functions
        if self.loss_fn is not None and self.num_labels == 1:
            logits = logits.view(-1).to(labels.dtype)
            loss = self.loss_fn(logits, labels.view(-1))
        # default hugging face loss function handling for all problems
        elif labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return loss

