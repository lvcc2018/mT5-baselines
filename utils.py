import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import os

LANGUAGE = ['en', 'zh', 'es', 'fr']

def dump_file(obj:list, file_path:str):
	open(file_path,'w').writelines([json.dumps(i, ensure_ascii=False)+'\n' for i in obj])

def load_file(file_path:str)->list:
	return [json.loads(i.strip()) for i in open(file_path).readlines()]

def load_txt_file(file_path:str):
	return [i.strip() for i in open(file_path).read().split('Result:') if i.strip()!='']
	

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default='ALL')
	parser.add_argument("--epoch_num", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--learning_rate", type=float, default=1e-5)
	parser.add_argument("--data_path", type=str, default="/home/lvcc/mT5-baselines/data/")
	parser.add_argument("--freeze_plm", type=bool, default=False)
	parser.add_argument("--model_path", type=str, default="google/mt5-small")
	parser.add_argument("--seed", type=int, default=666)
	parser.add_argument("--device", type=str, default='cuda:0')
	parser.add_argument("--shotnum", type=str, default='full')
	parser.add_argument("--utils", type=str, default='')
	args = parser.parse_args()
	return args



def get_dataset(args):
	file_path = os.path.join(args.data_path+args.dataset, args.shotnum)
	dataset = load_dataset('json', data_files={i:os.path.join(file_path, i+'.json') for i in ['train','valid','test']})
	return dataset