import json
from tqdm import tqdm
import os
import numpy as np
from transformers import BertTokenizerFast,AdamW
import torch
from tools.argparsetool import getparse
from tools.dataprocess import loadvocab,loaddata,loadschemas
from tools.dealdata import data_generator,SPO
from model.spomodel import ObjectModel,SubjectModel
from adv.adversarial import PGD

args = getparse().parse_args()
#选择device
device = torch.device(f'cuda:{args.GPUNUM}') if torch.cuda.is_available() else torch.device('cpu')

#加载数据
vocab = loadvocab(args.vocab_path)
train_data = loaddata(args.train_path)
dev_data = loaddata(args.dev_path)
idtopredicate,predicatetoid = loadschemas(args.schemas_path)

#加载初始化模型
if os.path.exists('graph_model.bin'):
    print('load_model')
    model = torch.load('graph_model.bin').to(device)
    subject_model = model.encoder
else:
    K = 5
    subject_model = SubjectModel.from_pretrained(args.bert_path)
    subject_model = subject_model.to(device)
    model = ObjectModel(subject_model,args.hidden)
    model = model.to(device)
    pgd = PGD(model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)

optim = AdamW(model.parameters(),lr=args.lr)
loss_fun = torch.nn.BCELoss()
model.train()
f1_total = 0
#处理加载数据
tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
train_loader = data_generator(train_data,tokenizer,idtopredicate,predicatetoid,8)

#训练模型
def train_fun(f1_total):
    train_loss = 0
    pbar = tqdm(train_loader)
    for step,batch in enumerate(pbar):
        optim.zero_grad()
        batch = batch[0]
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        subject_labels = batch[3].to(device)
        subject_ids = batch[2].to(device)
        object_labels = batch[4].to(device)
        subject_out,object_out = model(input_ids,subject_ids.float(),attention_mask)
        subject_out = subject_out * attention_mask.unsqueeze(-1)
        object_out = object_out * attention_mask.unsqueeze(-1).unsqueeze(-1)

        subject_loss = loss_fun(subject_out,subject_labels.float())
        object_loss = loss_fun(object_out,object_labels.float())

        loss = subject_loss + object_loss
        train_loss += loss.item()

        loss.backward()
        pgd.backup_grad()
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            subject_out, object_out = model(input_ids, subject_ids.float(), attention_mask)
            subject_out = subject_out * attention_mask.unsqueeze(-1)
            object_out = object_out * attention_mask.unsqueeze(-1).unsqueeze(-1)

            subject_loss_adv = loss_fun(subject_out, subject_labels.float())
            object_loss_adv = loss_fun(object_out, object_labels.float())

            loss_adv = subject_loss_adv + object_loss_adv
            loss_adv.backward()
        pgd.restore()
        optim.step()

        pbar.update()
        pbar.set_description(f'train_loss:{loss.item()}')


        if step % 500 == 0 and step != 0:
            with torch.no_grad():
                X,Y,Z = 1e-10,1e-10,1e-10
                pbar = tqdm()
                for data in dev_data:
                    spo = []
                    spo_text = data['text']
                    spo_ori = data['spo_list']
                    en = tokenizer(text=spo_text,return_tensors='pt')
                    _,subject_preds = subject_model(en.input_ids.to(device),en.attention_mask.to(device))
                    subject_preds = subject_preds.cpu().data.numpy()
                    start = np.where(subject_preds[0,:,0]>0.6)[0]
                    end = np.where(subject_preds[0,:,1]>0.5)[0]
                    subjects = []
                    for i in start:
                        j = end[end>=i]
                        if len(j) > 0:
                            j = j[0]
                            subjects.append((i,j))
                    # if subjects:
                    #     for s in subjects:
                    #         index = en.input_ids.cpu().data.numpy().squeeze(0)[s[0]:s[1]+1]
                    #         subject =''.join([vocab[i] for i in index])
                    #
                    #         _,object_preds = model(en.input_ids.to(device),torch.from_numpy(np.array([s])).float().to(device),en.attention_mask.to(device))
                    #         object_preds = object_preds.cpu().data.numpy()
                    #         for object_pred in object_preds:
                    #             start = np.where(object_pred[:,:,0] > 0.2)
                    #             end = np.where(object_pred[:,:,1]>0.2)
                    #
                    #             for _start,predicate1 in zip(*start):
                    #                 for _end,predicate2 in zip(*end):
                    #                     if _end>=_start and predicate1 == predicate2:
                    #                         index = en.input_ids.cpu().data.numpy().squeeze(0)[_start:_end+1]
                    #                         object = ''.join([vocab[i] for i in index ])
                    #                         predicate = idtopredicate[str(predicate1)]
                    #                         spo.append([subject,predicate,object])
                    #                         print([subject,predicate,object])
                    if subjects:
                        for s in subjects:
                            index = en.input_ids.cpu().data.numpy().squeeze(0)[s[0]:s[1] + 1]
                            subject = ''.join([vocab[i] for i in index])

                            _, object_preds = model(en.input_ids.to(device),
                                                    torch.from_numpy(np.array([s])).float().to(device),
                                                    en.attention_mask.to(device))
                            object_preds = object_preds.cpu().data.numpy()
                            for object_pred in object_preds:
                                start = np.where(object_pred[:, :, 0] > 0.3)
                                end = np.where(object_pred[:, :, 1] > 0.3)
                                for _start, predicate1 in zip(*start):
                                    for _end, predicate2 in zip(*end):
                                        if _start <= _end and predicate1 == predicate2:
                                            index = en.input_ids.cpu().data.numpy().squeeze(0)[_start:_end + 1]
                                            object = ''.join([vocab[i] for i in index])
                                            predicate = idtopredicate[str(predicate1)]
                                            spo.append([subject, predicate, object])



                    R = set([SPO(_spo) for _spo in spo])

                    T = set([SPO(_spo) for _spo in spo_ori])

                    X +=len(R&T)
                    Y +=len(R)
                    Z +=len(T)

                    f1,precision,recall = 2*X/(Y+Z),X/Y,X/Z
                    if f1 > f1_total:
                        f1_total = f1
                        torch.save(model,'graph_model.bin')

                    pbar.update()
                    pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))

                pbar.close()
                print('f1:',f1,'precision:',precision,'recall:',recall)
    return f1_total

#主程序

for epoch in range(20):
    f1_total = train_fun(f1_total)
print(f1_total)




















