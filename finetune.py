import enum
from sched import scheduler
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup
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
    result = tokenizer(examples['source'], padding="max_length", max_length=max_source_length, truncation=True)
    result['labels'] = tokenizer(examples['target'], padding="max_length", max_length=max_target_length, truncation=True).input_ids
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

total_steps = len(dataloader['train']) * args.epoch_num

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
loss_func = torch.nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=0.1*total_steps, num_training_steps=total_steps)

writer = SummaryWriter('logs/mT5_XNLI_lr1e-5')

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
        scheduler.step()
        optimizer.zero_grad()
        iter_loss = loss.item()
        epoch_loss += iter_loss
        writer.add_scalar('train/loss', iter_loss, epoch*len(dataloader['train']) + idx)
    epoch_loss /= len(dataloader['train'])
    print('--------- Epoch {} Loss {} ---------'.format(epoch, epoch_loss))

def main():
    model.to(args.device)
    min_loss = 1e10
    for epoch in range(args.epoch_num):
        train(epoch)
        torch.save(model.state_dict(), f'./checkpoints/mT5_{args.dataset}_epoch{epoch}.pt')
        loss_eval = evaluate(epoch)
        if loss_eval < min_loss:
            torch.save(model.state_dict(), f'./checkpoints/mT5_{args.dataset}_best_eval.pt')
            print("Better checkpoint saved")
            min_loss = loss_eval

def evaluate(epoch):
    with torch.no_grad():
        model.eval()
        epoch_loss = 0.
        for idx, data in enumerate(tqdm(dataloader['valid'], desc='Evaluating Epoch {}'.format(epoch))):
            input_ids, attention_mask, labels = data
            input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            epoch_loss += loss.item()
            writer.add_scalar('valid/loss', epoch_loss, epoch*len(dataloader['valid'] + idx))
        epoch_loss /= len(dataloader['valid'])
        print('--------- Eval Epoch {} Loss {} ---------'.format(epoch, epoch_loss))
    return epoch_loss


if __name__ == '__main__':
    main()
