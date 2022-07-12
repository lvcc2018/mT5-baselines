import enum
from sched import scheduler
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup
import torch
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import *
from collections import OrderedDict

class SoftEmbedding(torch.nn.Module):
    def __init__(self, raw_embedding, n_tokens, lang):
        super().__init__()
        self.n_tokens = n_tokens
        self.lang = lang
        self.raw_embedding = raw_embedding
        self.soft_embedding = torch.nn.Embedding(self.n_tokens * len(self.lang), self.raw_embedding.weight.shape[-1])
        self.prefix_token = {}
        self.token2idx = {}
        for idx, i in enumerate(self.lang):
            self.prefix_token[i] = '<standfor' + str(i) + '>'
            self.token2idx[self.prefix_token[i]] = idx
    def forward(self, tokens):
        lang = []
        if tokens.shape[-1] >= self.n_tokens+3:
            if tokens[0, 3] >= 250100:
                prefix_embedding = self.raw_embedding(tokens[:, :3])
                input_embedding = self.raw_embedding(tokens[:, self.n_tokens+3:])
                for i in tokens[:, 3]:
                    lang.append(list(range(self.n_tokens*(i-250100), self.n_tokens*(i-250100)+self.n_tokens)))
                learned_embedding = self.soft_embedding(torch.IntTensor(lang).to(args.device))
                return torch.cat((prefix_embedding, learned_embedding, input_embedding), 1)
            else:
                return self.raw_embedding(tokens[:])
        else:
            return self.raw_embedding(tokens[:])

args = get_args()

dataset = get_dataset(args)
# dataset = get_all_dataset(args)

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
se = SoftEmbedding(model.get_input_embeddings(), 10, ['en', 'zh', 'hi', 'es', 'fr', 'ar'])
model.set_input_embeddings(se)
old_state_dict = model.state_dict()
state_dict = torch.load("/home/lvcc/mT5-baselines/pretrained/dict-10/base-300000.pt", map_location=torch.device('cpu'))

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
tokenizer.add_special_tokens({'additional_special_tokens': [i for i in se.prefix_token.values()]})
tokenizer.add_special_tokens({'additional_special_tokens': ['<lang>', '<ctxt>', '<task>']})

max_source_length = {'XLSUM':1024,'ALL':1024}[args.dataset]
max_target_length = {'XLSUM':128,'ALL':128}[args.dataset]

def preprocess(examples):
    result = tokenizer('<lang>:'+se.prefix_token[examples['language']]*10+';<ctxt>:'+examples['source'], padding="max_length", max_length=max_source_length, truncation=True)
    result['labels'] = tokenizer(examples['target'], padding="max_length", max_length=max_target_length, truncation=True).input_ids
    return result


def preprocess_test(examples):
    result = tokenizer('<lang>:'+se.prefix_token[examples['language']]*10+';<ctxt>:'+examples['source'], padding="max_length", max_length=max_source_length, truncation=True)
    result['labels'] = examples['target']
    return result

def preprocess_task(examples):
    result = tokenizer('<lang>:'+se.prefix_token[examples['language']]*10+';<task>:'+examples['task']+';<ctxt>:'+examples['source'], padding="max_length", max_length=max_source_length, truncation=True)
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
# dataset['test'] = dataset['test'].filter(lambda example: example['language'] == 'zh')
dataset['test'] = dataset['test'].map(preprocess_task, desc='Tokenizing Test Dataset')
dataloader = {split: DataLoader(dataset[split], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn) for split in ['train', 'valid']}
dataloader['test'] = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

total_steps = len(dataloader['train']) * args.epoch_num

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
loss_func = torch.nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

writer = SummaryWriter('/home/lvcc/mT5-baselines/logs/mT5_{}_{}_dict10_task'.format(args.dataset, args.learning_rate))

def train(epoch):
    epoch_loss = 0.
    iter_loss = 0.
    with tqdm(total=len(dataloader['train'])) as t:
        for idx, data in enumerate(dataloader['train']):
            t.set_description('Training Epoch {}'.format(epoch))
            model.train()
            input_ids, attention_mask, labels = data
            input_ids, attention_mask, labels = input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            iter_loss = loss.item()
            t.set_postfix(loss=iter_loss)
            t.update(1)
            writer.add_scalar('train/loss', iter_loss, epoch*len(dataloader['train']) + idx)

def main():
    model.to(args.device)
    min_loss = 1e10
    for epoch in range(args.epoch_num):
        train(epoch)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'/home/lvcc/mT5-baselines/checkpoints/mT5_{args.dataset}_{args.learning_rate}_dict10_task_epoch{epoch}.pt')
        loss_eval = evaluate(epoch)
        if loss_eval < min_loss:
            torch.save(model.state_dict(), f'/home/lvcc/mT5-baselines/checkpoints/mT5_{args.dataset}_{args.learning_rate}_dict10_task_best_eval.pt')
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
    state_dict = torch.load("./checkpoints/mT5_MARC_dict_best_eval.pt", map_location=torch.device('cpu'))
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
    # f = open("notpure_all.json", 'w')
    # json.dump(temp, f, ensure_ascii=False)

def inference(args):
    f = open(f'/home/lvcc/mT5-baselines/result/mT5_{args.dataset}_{args.learning_rate}_dict10_task_best_eval.txt','w')
    model.load_state_dict(torch.load(f'/home/lvcc/mT5-baselines/checkpoints/mT5_ALL_1e-05_dict10_task_best_eval.pt'))
    model.to(args.device)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader['test'], desc='Inferencing...')):
            model.eval()
            input_ids, attention_mask, labels = data
            input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)
            res = model.generate(inputs=input_ids, max_length=max_target_length, top_p=0.9, temperature=0.9)
            res = tokenizer.batch_decode(res, skip_special_tokens=True)
            for r in res:
                f.write("Result:")
                f.write(r)
                f.write('\n')
                f.flush()

if __name__ == '__main__':
    # generate(0)
    # main()
    print(args)
    inference(args)
