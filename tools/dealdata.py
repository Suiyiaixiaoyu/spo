import numpy as np
import torch

class SPO(tuple):
    def __init__(self,spo):
        self.spox=(
            spo[0],
            spo[1],
            spo[2]
        )

    def __hash__(self):
        return self.spox.__hash__()
    def __eq__(self, spo):
        return self.spox == spo.spox

def search(patten,sequence):
    n = len(patten)
    for i in range(len(sequence)):
        if sequence[i:i+n] == patten:
            return i
    return -1

def sequence_padding(inputs,length=None ,padding = 0,mode = 'post'):
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0,0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0,length-len(x))
        elif mode == 'pre':
            pad_width[0] = (length-len(x),0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre"')
        x =np.pad(x,pad_width,'constant',constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def data_generator(data,tokenizer,idtopredicate,predicatetoid,batch_size=3,):
    batch_input_ids,batch_attention_mask = [],[]
    batch_subject_labels,batch_subject_ids,batch_object_labels = [],[],[]
    texts = []
    for i,d in enumerate(data):
        text = d['text']
        texts.append(text)
        encoding = tokenizer(text = text)
        input_ids,attention_mask = encoding.input_ids,encoding.attention_mask
        spoes = {}
        for s,p,o in d['spo_list']:
            s_encoding = tokenizer(text=s).input_ids[1:-1]
            o_encoding = tokenizer(text=o).input_ids[1:-1]

            s_idx = search(s_encoding,input_ids)
            o_idx = search(o_encoding,input_ids)

            p = predicatetoid[p]

            if s_idx !=-1 and o_idx !=-1:
                s = (s_idx,s_idx+len(s_encoding)-1)
                o = (o_idx,o_idx+len(o_encoding)-1,p)
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)

        if spoes:
            subject_labels = np.zeros((len(input_ids),2))
            for s in spoes:
                subject_labels[s[0],0] = 1
                subject_labels[s[1],1] = 1
            start ,end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            # end = np.random.choice(end[end>=start][0])
            end = end[end >= start][0]
            subject_ids = (start,end)
            object_labels = np.zeros((len(input_ids),len(predicatetoid),2))
            for o in spoes.get(subject_ids,[]):
                object_labels[o[0],o[2],0] = 1
                object_labels[o[1],o[2],1] = 1
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_subject_labels.append(subject_labels)
            batch_subject_ids.append(subject_ids)
            batch_object_labels.append(object_labels)
            if len(batch_subject_labels) == batch_size or i == len(data) - 1:
                batch_input_ids = sequence_padding(batch_input_ids)
                batch_attention_mask = sequence_padding(batch_attention_mask)
                batch_subject_labels = sequence_padding(batch_subject_labels)
                batch_subject_ids = np.array(batch_subject_ids)
                batch_object_labels = sequence_padding(batch_object_labels)
                yield [
                    torch.from_numpy(batch_input_ids).long(),torch.from_numpy(batch_attention_mask).long(),
                    torch.from_numpy(batch_subject_ids),torch.from_numpy(batch_subject_labels),
                    torch.from_numpy(batch_object_labels)

                ],None
                batch_input_ids,batch_attention_mask = [],[]
                batch_subject_labels,batch_subject_ids,batch_object_labels = [],[],[]













