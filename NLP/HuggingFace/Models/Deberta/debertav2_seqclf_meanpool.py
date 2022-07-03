
from debertav2_seqclf_base import DebertaV2ForSeqClfBase
from typing import Optional, Union, Tuple
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

class DebertaV2ForSeqClfMeanPooling(DebertaV2ForSeqClfBase):    
    def __init__(self, config):
        super().__init__(config)
    
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
        # copy the 2d attention mask [batch_size, seq_len] hidden_size times in the third dimension (hidden state)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # [batch_size, seq_len, hidden_size ]
        # Each hidden state is a tensor of dimension [batch_size, seq_len] and we have hidden_size number of such hidden state
        # Of these seq_len columns in each hidden state only those need to be taken into account for which attention_mask = 1.  
        # Doing an element wise multiplication of the 2d attention mask [batch_size, seq_len] with the corresponding 2d hidden state 
        # [batch_size, seq_len] gives hidden state with only the non-padded columns. Sum this hidden state along dimension 1 
        # ( the dimension of sequence length) to get the sum_embeddings [batch_size, hidden_size] 
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # The sum mask is a value between 0 to 256 signifying the number of unpadded columns for that hidden state
        sum_mask = input_mask_expanded.sum(1)
        # [batch_size, hidden_size]
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        # element wise division 
        mean_embeddings = sum_embeddings / sum_mask
        # [batch_size, hidden_size]
        logits = self.classifier(mean_embeddings)
        loss = self.get_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )