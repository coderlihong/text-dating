import os

import torch
from transformers import BertTokenizer, AutoConfig, BertForMaskedLM, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments,BertConfig

file_list = os.listdir('../data/time_data/')
embedding_dict = {}
for i in file_list:
    name = i.split('.')[0]
    tokenizer = BertTokenizer.from_pretrained('bert-ancient-chinese')
    train_dataset = LineByLineTextDataset(file_path=f'../data/time_data/{name}.txt', tokenizer=tokenizer, block_size=40)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # val_dataset = LineByLineTextDataset(file_path='./data/val_bert.txt', tokenizer=tokenizer, block_size=128)

    model = BertForMaskedLM.from_pretrained('bert-ancient-chinese')

    training_args = TrainingArguments(
        output_dir='./single_train_weight/',
        num_train_epochs=10,
        per_device_train_batch_size=256,
        warmup_steps=500,
        report_to=['none'],
        weight_decay=0.01,
        do_train=True,
        logging_steps=1,
        learning_rate=3e-5,
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        save_strategy='no',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,

    )

    trainer.train()

    torch.save(model.bert.embeddings.word_embeddings.weight,f'{name}.pt')

