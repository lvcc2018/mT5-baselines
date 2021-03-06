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

max_source_length = 1024
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
dataloader = {split: DataLoader(dataset[split], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) for split in ['train', 'valid']}
dataloader['test'] = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

total_steps = len(dataloader['train']) * args.epoch_num

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
loss_func = torch.nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

writer = SummaryWriter('logs/mT5_{}_'.format(args.dataset)+str(args.learning_rate))

def train(epoch):
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
        writer.add_scalar('train/loss', iter_loss, epoch*len(dataloader['train']) + idx)
    print('--------- Epoch {} Loss {} ---------'.format(epoch, iter_loss))

def main():
    model.to(args.device)
    min_loss = 1e10
    for epoch in range(args.epoch_num):
        train(epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'./checkpoints/mT5_{args.learning_rate}_{args.dataset}_epoch{epoch}.pt')
        loss_eval = evaluate(epoch)
        if loss_eval < min_loss:
            torch.save(model.state_dict(), f'./checkpoints/mT5_{args.learning_rate}_{args.dataset}_best_eval_pro.pt')
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
            writer.add_scalar('valid/loss', loss.item(), epoch*len(dataloader['valid']) + idx)
        epoch_loss /= len(dataloader['valid'])
        print('--------- Eval Epoch {} Loss {} ---------'.format(epoch, epoch_loss))
    return epoch_loss


def inference(args):
    f = open(f'./result/mT5_{args.learning_rate}_{args.dataset}_best_eval_test.txt','w')
    model.load_state_dict(torch.load(f'/home/lvcc/mT5-baselines/checkpoints/mT5_1e-05_ALL_best_eval_pro.pt'))
    model.to(args.device)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader['test'], desc='Inferencing...')):
            model.eval()
            input_ids, attention_mask, labels = data
            input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
            res = model.generate(inputs=input_ids, max_length=128, top_p=0.9, temperature=0.9)
            res = tokenizer.batch_decode(res, skip_special_tokens=True)
            for r in res:
                f.write("Result:")
                f.write(r)
                f.write('\n')
                f.flush()




if __name__ == '__main__':
    print(args)
    # main()
    inference(args)
    
