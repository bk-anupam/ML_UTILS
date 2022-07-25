
from debertav2_seqclf_base import DebertaV2ForSeqClfBase
from typing import Optional, Union, Tuple
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

'''
Default deberta-v2 sequence classification head, uses the pooler output (last layer hidden state
of the first token [CLS] further processed by a classifier)
'''
class DebertaV2ForSeqClf(DebertaV2ForSeqClfBase):
    def __init__(self, config, loss_type: str=None):
        super().__init__(config, loss_type=loss_type)

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

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
        # sequence of hidden states from the last layer [batch_size, seq_len, hidden_size]
        encoder_layer = outputs[0]
        # The pooling layer takes the hidden state corresponding to the first token of the
        # sequence , the [CLS] token [:, 0, :], we effectively reduce the 3d tensor to
        # a 2d tensor with dimensions [batch_size, hidden_size]
        pooled_output = self.pooler(encoder_layer)
        #[batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        loss = self.get_loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

