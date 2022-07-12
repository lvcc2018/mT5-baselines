import enum
from sched import scheduler
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup
import torch
import evaluate as eval
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import *


args = get_args()

dataset = get_dataset(args)
# dataset = get_all_dataset(args)

tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

max_source_length = 512
max_target_length = 128

def preprocess(examples):
    result = tokenizer(examples['source'], padding="max_length", max_length=max_source_length, truncation=True)
    result['labels'] = tokenizer(examples['target'], padding="max_length", max_length=max_target_length, truncation=True).input_ids
    return result

def preprocess_test(examples):
    result = tokenizer(examples['source'], padding="max_length", max_length=max_source_length, truncation=True)
    result['labels'] = examples['target']
    return result

def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])
        labels.append(data['labels'])
    if not isinstance(labels[0], str):
        labels = torch.tensor(labels)
        labels[labels == tokenizer.pad_token_id] = -100
    input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)
    return input_ids, attention_mask, labels

dataset['train'] = dataset['train'].map(preprocess, desc='Tokenizing Train Dataset')
dataset['valid'] = dataset['valid'].map(preprocess, desc='Tokenizing Valid Dataset')
dataset['test'] = dataset['test'].filter(lambda example: example['language'] == 'fr')
dataset['test'] = dataset['test'].map(preprocess_test, desc='Tokenizing Test Dataset')
dataloader = {split: DataLoader(dataset[split], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) for split in ['train', 'valid', 'test']}

total_steps = len(dataloader['train']) * args.epoch_num

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
loss_func = torch.nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

writer = SummaryWriter('logs/mT5_{}'.format(args.dataset))

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
            iter_loss = loss.item()
            epoch_loss += iter_loss
            writer.add_scalar('valid/loss', iter_loss, epoch*len(dataloader['valid']) + idx)
        epoch_loss /= len(dataloader['valid'])
        print('--------- Eval Epoch {} Loss {} ---------'.format(epoch, epoch_loss))
    return epoch_loss

def generate(epoch):
    state_dict = torch.load("./checkpoints/mT5_MARC_best_eval.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(args.device)
    allpreds = []
    alllabels = []
    with torch.no_grad():
        model.eval()
        for idx, data in enumerate(tqdm(dataloader['test'], desc='Testing Epoch {}'.format(epoch))):
            input_ids, attention_mask, labels = data
            input_ids, attention_mask= input_ids.to(args.device), attention_mask.to(args.device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            alllabels.extend(labels)
            allpreds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print(acc)
    # temp = {'target': alllabels, 'prediction':allpreds}
    # f = open("pure_all.json", 'w')
    # json.dump(temp, f, ensure_ascii=False)

def mybleu():
    bleu = eval.load("bleu")
    gg = json.load(open('notpure.json', 'r'))
    results = bleu.compute(predictions=gg['prediction'], references=gg['target'], tokenizer=tokenizer.tokenize, max_order=1)
    # rouge = eval.load("rouge")
    # results = rouge.compute(predictions=gg['prediction'], references=gg['target'])
    print(results)



if __name__ == '__main__':
    # mybleu()
    generate(0)
    # main()
