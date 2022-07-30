from debertav2_seqclf_base import DebertaV2ForSeqClfBase
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from hf_model_head_utils import AttentionHead

class DebertaV2ForSeqClfConcatLastFour(DebertaV2ForSeqClfBase):    
    '''
    Deberta V2 sequence classifier using representations from the last four encoder layers. 
    The output of each encoder layer along each token's path can be used as a feature
    representing that token. For each token concatenating the features returned by the last
    four encoder layers gives the best representation.
    '''
    def __init__(self, config, loss_type: str=None):
        super().__init__(config, loss_type=loss_type)
        # default hidden_size = 1536 in DebertaV2 config        
        self.fc = nn.Linear(config.hidden_size * 4, 1)
    
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
        # output_hidden_states parameter needs to be passed as True for output from each encoder layer to be
        # returned (This has already been set in model config)
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # the second output term contains hidden state from each encoder layer
        all_hidden_states = torch.stack(outputs[1])
        # [num_layers, batch_size, seq_len, hidden_size] 
        concatenate_pooling = torch.cat((
            all_hidden_states[-1], 
            all_hidden_states[-2], 
            all_hidden_states[-3], 
            all_hidden_states[-4]), -1
        )
        # [batch_size, seq_len, 4 * hidden_size]
        # Take the representation corresponding to the first token [CLS]
        concatenate_pooling = concatenate_pooling[:, 0]
        # [batch_size, 4 * hidden_size]
        logits = self.fc(concatenate_pooling)
        # [batch_size, 1]        
        loss = self.get_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )