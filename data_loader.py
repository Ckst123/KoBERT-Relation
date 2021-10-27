from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

checkpoint = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    


def load_data():
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'})
    print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example["headline"], truncation=True)

    def trim(example):
        example['headline'] = example['headline'].strip()
        return example

    tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ['headline', 'relation']
    )
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader

if __name__ == '__main__':
    load_data()