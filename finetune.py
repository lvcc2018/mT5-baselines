import enum
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import *


args = get_args()

dataset = get_dataset(args)

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

max_source_length = 512
max_target_length = 128

def preprocess(examples):
    result = tokenizer(examples['source'], padding="max_length", max_length=max_source_length)
    result['labels'] = tokenizer(examples['target'], padding="max_length", max_length=max_target_length).input_ids
    return result

def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])
        labels.append(data['labels'])
    input_ids, attention_mask, labels = torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, attention_mask, labels

dataset['train'] = dataset['train'].map(preprocess, desc='Tokenizing Train Dataset')
dataset['valid'] = dataset['valid'].map(preprocess, desc='Tokenizing Valid Dataset')
dataset['test'] = dataset['test'].map(preprocess, desc='Tokenizing Test Dataset')
dataloader = {split: DataLoader(dataset[split], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) for split in ['train', 'valid', 'test']}

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_func = torch.nn.CrossEntropyLoss()

writer = SummaryWriter('logs/mT5_{}'.format(args.data_path))

def train(epoch):
    epoch_loss = 0.
    iter_loss = 0.
    for idx, data in enumerate(tqdm(dataloader['train'], desc='Training Epoch {}'.format(epoch))):
        model.train()
        input_ids, attention_mask, labels = data
        input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()
        iter_loss = loss.item()
        epoch_loss += iter_loss()
        writer.add_scalar('train/loss', iter_loss, epoch*len(dataloader['train'] + idx))
    epoch_loss /= len(dataloader['train'])
    print('--------- Epoch {} Loss {} ---------'.format(epoch, epoch_loss))

def main():
    model.to(args.device)
    for epoch in range(args.epoch_num):
        train(epoch)
        torch.save(model.state_dict(), f'./checkpoints/mT5_{args.dataset}_epoch{epoch}.pt')


if __name__ == '__main__':
    main()
