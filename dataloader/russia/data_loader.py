from dataclasses import dataclass
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import sys
sys.path.append('/Users/sunlin/Documents/workdir/ieer/infromationExtractorInBFQB-main/')
from utils import HParam
import csv
import os
from transformers import AutoTokenizer
import numpy as np
def pad(ids, pad_id, max_length):
    if len(ids) > max_length:
        return ids[:max_length]
    return ids + [pad_id] * (max_length - len(ids))

class BaseDataset(Dataset):
    '''
    dataset for information extractor
    '''

    def __init__(self, args,tokenizer,data_set,add_special_tokens=True,do_sth='do_train'):
        super().__init__()
        self.args_data = args
        self.tokenizer = tokenizer
        self.do_sth = do_sth
        #self.data_size = os.path.getsize(args.data_path)/1024/1024/1024
        self.data = data_set
        self.max_seq_length = args.max_seq_length
        #self.max_seq_length =-1
        self.add_special_tokens = add_special_tokens


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index])

    def data_parse(self, line):
        """
        解析不同格式的数据
        """
        #dic = csv.reader(line,delimiter=',')
        return line

    def encode(self, item):
        """
        将数据转换成模型训练的输入
        """
        #print(self.do_sth)
        if self.do_sth == 'do_train':
            single_data = self._trainSet(item)
        elif self.do_sth == 'do_test':
            single_data = self._testSet(item)
        return single_data

    def _trainSet(self,item):
# time, mid, blue, red,trible, discription
        # conver to ids
        #print(item.keys())
        #print(item['data']['prompt'],'\n--------------------------annotations-----------------\n',item['annotations'][0]['result'])
        r_qb = item['notes_dst']

        #label_map = np.zeros((self.args_data.num_labels,self.max_seq_length))
        label_map = np.zeros((self.max_seq_length))
        
        #input
        prompt_ids = self.tokenizer.encode_plus(item['notes_dst'],max_length=self.max_seq_length,
                                                padding='max_length',
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
        
        #unknow
        #time
        time_position = item['notes_dst'].find(item['event_date_dst'])
        label_map[time_position]=1
        label_map[(time_position+1):time_position+len(item['event_date_dst'])-1]=2
        label_map[time_position+len(item['event_date_dst'])-1]=3

        #action1
        time_position = item['notes_dst'].find(item['actor1_dst'])
        #print(time_position)
        label_map[time_position]=4
        label_map[(time_position+1):time_position+len(item['actor1_dst'])-1]=5
        label_map[time_position+len(item['actor1_dst'])-1]=6

        #action2
        time_position = item['notes_dst'].find(item['actor2_dst'])
        label_map[time_position]=7
        label_map[(time_position+1):time_position+len(item['actor2_dst'])-1]=8
        label_map[time_position+len(item['actor2_dst'])-1]=9

        #location
        time_position = item['notes_dst'].find(item['location_dst'])
        label_map[time_position]=10
        label_map[(time_position+1):time_position+len(item['location_dst'])-1]=11
        label_map[time_position+len(item['location_dst'])-1]=12

        return  {"input_ids": prompt_ids['input_ids'].squeeze().clone().detach(),
                 "attention_mask": prompt_ids['attention_mask'].squeeze().clone().detach(), 
                 #"position_ids": torch.arange(0, self.max_seq_length).clone().detach(),
                 'text': r_qb,
                 "labels":  torch.tensor(label_map,dtype=torch.int).clone().detach()}

    def _testSet(self,item):
# time, mid, blue, red,trible, discription
        # conver to ids
        r_qb = item['data'][0]
        #print(r_qb)
        
        prompt_ids = self.tokenizer.encode_plus(r_qb,max_length=self.max_seq_length,
                                                padding='max_length',
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')

        return  {"input_ids": prompt_ids['input_ids'].squeeze().clone().detach(),
                 "attention_mask": prompt_ids['attention_mask'].squeeze().clone().detach(), 
                 #"position_ids": torch.arange(0, self.max_seq_length).clone().detach(),
                 'text': r_qb,
                 "labels": torch.zeros(5)}
        
    
def dataLoaderBase(tokenizer,args_data,do_sth):
        if do_sth == 'do_train':
            datasets = load_dataset(args_data.raw_file_type, data_files={'train':args_data.train_file,
                                                                        'validation':args_data.valid_file,
                                                                        'debug':args_data.debug_file,
                                                                                                            })
            train_set = datasets['debug'] if args_data.debug_mode  else datasets['train']
            #print(datasets['train'][:4])
            train_dataset = BaseDataset(args_data,tokenizer,train_set,True,'do_train')
            valid_dataset = BaseDataset(args_data,tokenizer,train_set,True,'do_train')
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=args_data.train_batchsize,
                num_workers=args_data.num_workers,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=args_data.valid_batchsize,
                shuffle=False,
                num_workers=args_data.num_workers,
                pin_memory=False,
            )
            return train_loader, valid_loader
        elif do_sth == 'do_test':
            print('Reading datasets in test.json')
            datasets = load_dataset(args_data.raw_file_type, data_files={'test':args_data.test_file,
                                                                                                            })
            #print(datasets)
            test_dataset = BaseDataset(args_data,tokenizer,datasets['test'],'do_test')
            test_loader = DataLoader(
                test_dataset,
                batch_size=args_data.test_batchsize,
                shuffle=False,
                #num_workers=args_data.num_workers,
                pin_memory=False,
            )
            return test_loader


if __name__ == '__main__':
    hp = HParam('/Users/sunlin/Documents/workdir/ieer/infromationExtractorInBFQB-main/config/default.yaml')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    data_module  = BaseDataModule(tokenizer,hp.data)
    with open('demo.csv', 'w') as file:
        writer = csv.writer(file)
        for i in data_module.train_dataset:
            print(i['input_ids'].shape,i['labels'].shape)
            label = i['labels']
            writer.writerows([i['text']])
            writer.writerows(np.array(i['labels'].data))
            break

        