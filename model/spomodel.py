from transformers import BertModel,BertPreTrainedModel
import torch.nn as nn
import torch

class SubjectModel(BertPreTrainedModel):
    def __init__(self,config):
        super(SubjectModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size,2)

    def forward(self,input_ids,attention_mask):
        output = self.bert(input_ids=input_ids,attention_mask = attention_mask)
        subject_out = self.dense(output[0])
        subject_out = torch.sigmoid(subject_out)
        return output[0],subject_out

class ObjectModel(nn.Module):
    def __init__(self,SubjectModel,hidden):
        super(ObjectModel, self).__init__()
        self.enconder = SubjectModel
        self.denseSubject = nn.Linear(2,hidden)
        self.denseobject  = nn.Linear(hidden,49*2)

    def forward(self,input_ids,subject_position,attention_mask):
        output,subject_out = self.enconder(input_ids,attention_mask)
        subject_position = self.denseSubject(subject_position).unsqueeze(1)
        object_out = subject_position+output
        object_out = self.denseobject(object_out)
        object_out = torch.reshape(object_out,(object_out.shape[0],object_out.shape[1],49,2))
        object_out = torch.sigmoid(object_out)
        object_out = torch.pow(object_out,4)
        return subject_out,object_out




