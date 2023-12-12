import pandas as pd
import torch
import torch.nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR

from model import Mymodel, HAN, Projection
from transformers import AutoTokenizer
from dataset import getdata, get_weight, load_data
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score,recall_score
from tqdm import tqdm
class Config():
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-ancient-chinese")
        self.embedding_size = 768
        self.lr = 1e-5
        self.class_num = 12
        self.max_sentence_len = 30
        self.max_tokens_len = 32
        self.batch_size = 8
        self.epochs = 100

        self.base_data_path = 'corpus_data'
        self.gru_size = 100
        self.cls_layers =6
        self.mlm_layers = 6
        self.bert_path = ''

k=None
print('k=',k)
config = Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
projection = Projection(config.tokenizer, config,device)

han = HAN(config,config.tokenizer,device)

model = Mymodel(projection, han, config.tokenizer, config, device=device)
# model.load_state_dict(torch.load('model123.pth'))
model.to(device)
train_loader, val_loader, test_loader, weight, label2idx,word_freq = load_data(config, config.tokenizer,k)
print(label2idx)
opt = optim.AdamW(model.parameters(), lr=config.lr,weight_decay=1e-4)
scheduler = StepLR(opt, step_size=10, gamma=0.5)
critertion_cls = nn.CrossEntropyLoss()
critertion_proj = nn.MSELoss()
critertion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
scaler = GradScaler()
from torch.distributions.normal import Normal


def train():

    bestf1 = 0
    for epoch in range(config.epochs):
        epoch_loss = 0
        train_labels = []
        train_pred = []
        for iter, batch in enumerate(tqdm(train_loader)):

            batch = [item.to(device) for item in batch]
            document_encode, sentence_attention, target, label, mlm_lables = batch
            opt.zero_grad()
            with autocast():
                output, projection, mlm_output = model(document_encode, sentence_attention)
                loss2 = critertion_proj(projection.view(-1, projection.size(3)), target.view(-1, target.size(3)))

                loss1 = critertion_cls(output, label)
                loss_mlm = critertion_mlm(mlm_output.view(-1,mlm_output.size(3)),mlm_lables.view(-1))

                loss = loss1+loss_mlm+loss2

            pred = output.argmax(dim=-1)

            train_labels.extend(label.cpu().detach().tolist())
            train_pred.extend(pred.cpu().detach().tolist())
            if iter == 1:
                print(loss)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()



            epoch_loss += loss.item()

        scheduler.step()

        train_labels, train_pred = np.array(train_labels), np.array(train_pred)
        acc = (train_labels == train_pred).sum() / len(train_labels)
        print(f"epoch:{epoch},epoch_loss:{epoch_loss / len(train_loader)}, train_acc :{acc:.4f}")
        val_labels = []
        val_pred = []
        for batch in val_loader:
            batch = [item.to(device) for item in batch]
            document_encode, sentence_attention, target, label  = batch

            with torch.no_grad():
                output, projection,mlm_output= model(document_encode, sentence_attention)
            pred = output.argmax(dim=-1)
            val_labels.extend(label.cpu().detach().tolist())
            val_pred.extend(pred.cpu().detach().tolist())
        val_labels, val_pred = np.array(val_labels), np.array(val_pred)
        acc = (val_labels == val_pred).sum() / len(val_pred)
        f1 = f1_score(val_labels,val_pred,average='macro')
        print(f" val_acc :{acc:.4f}, f1:{f1:.4f}")
        if bestf1<f1:
            bestf1 = f1
            torch.save(model.state_dict(),'model_k1.pth')


def test():
    model.load_state_dict(torch.load('model123.pth'))
    val_labels = []
    val_pred = []
    val_raw_sentence = []
    for batch in test_loader:
        batch = [item.to(device) for item in batch]
        document_encode, sentence_attention, target, label = batch

        with torch.no_grad():
            output, projection, mlm_output = model(document_encode, sentence_attention)
        pred = output.argmax(dim=-1)
        val_labels.extend(label.cpu().detach().tolist())
        val_pred.extend(pred.cpu().detach().tolist())
        document_encode = document_encode.view(document_encode.size(0),-1)
        decode_sentences = config.tokenizer.batch_decode(document_encode)
        decode_sentences = [''.join(item) for item in decode_sentences]
        val_raw_sentence.extend(decode_sentences)
    val_labels, val_pred = np.array(val_labels), np.array(val_pred)
    acc = (val_labels == val_pred).sum() / len(val_pred)
    recall = recall_score(val_labels, val_pred, average='macro')
    f1 = f1_score(val_labels, val_pred, average='macro')
    print(f" val_acc :{acc:.4f}, f1:{f1:.4f},recall:{recall}")
    new_val_labels = []
    new_val_pred = []
    new_val_raw_sentence = []
    for i,j,k in zip(val_labels,val_pred,val_raw_sentence):
        if i !=j:
            new_val_labels.append(i)
            new_val_pred.append(j)
            new_val_raw_sentence.append(k)

    pd.DataFrame({
        'pred':new_val_pred,
        'true':new_val_labels,
        'raw':new_val_raw_sentence
    }).to_excel('res_pred.xlsx',index=False)


if __name__ == '__main__':
    train()
    #
    test()