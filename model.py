import torch
import torch.nn as nn
from transformers import BertModel

class BERTClass(nn.Module):
    def __init__(self,num_labels = 298,token_len = 128,model_name = 'bert-base-cased',device = 'cuda:0'):
        super(BERTClass, self).__init__()
        self.token_len = token_len
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.dropout = nn.Dropout(0.3)
        self.last_layer = nn.Linear(768*token_len, num_labels)
    def forward(self,inputs):
        input_ids,token_type_ids,attention_mask = inputs.input_ids,inputs.token_type_ids,inputs.attention_mask
        input_ids,token_type_ids,attention_mask =  input_ids.squeeze(1),token_type_ids.squeeze(1),attention_mask.squeeze(1)
        output = self.model(input_ids,token_type_ids,attention_mask).last_hidden_state
        output = self.dropout(output)
        output = output.reshape(-1,self.token_len*768)
        return self.last_layer(output)

