from debertav2_seqclf_base import DebertaV2ForSeqClfBase
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from hf_model_head_utils import AttentionHead

class DebertaV2ForSeqClfAttention(DebertaV2ForSeqClfBase):    
    '''
    Deberta V2 sequence classifier using attention head. 
    '''
    def __init__(self, config, loss_type: str=None):
        super().__init__(config, loss_type=loss_type)
        # default hidden_size = 1536 in DebertaV2 config        
        self.attention = AttentionHead(
            hidden_size = config.hidden_size
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # hidden state from the last layer [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs[0]        
        context_vector = self.attention(last_hidden_state, attention_mask)
        # [batch_size, hidden_size]             
        # convert the context vector to prediction score   
        logits = self.classifier(context_vector)
        # [batch_size, 1]
        loss = self.get_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )