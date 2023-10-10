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
        relations=['属于','攻击','击毁','探测','成员']
        #print(item.keys())
        #print(item['data']['prompt'],'\n--------------------------annotations-----------------\n',item['annotations'][0]['result'])
        r_qb = item['data']['prompt'][0]

        result= item['annotations'][0]['result']
        label_map = np.zeros((self.max_seq_length+len(relations)+3*2*2,self.max_seq_length))
        end_map,start_map = {},{}
        for a_result in result:
            a_result_value = a_result['value']
            # 实例标定
            if a_result_value:
                if a_result_value['labels'][0] == '对象':
                    # 实例
                    label_map[0][a_result_value['start']] = 1
                    label_map[4][a_result_value['end']-1] = 1
                    if(len(a_result_value['text'])>2):
                        label_map[2][(a_result_value['start']+1):a_result_value['end']-1] = 1
                    start_map[a_result['id']]=a_result_value['start']

                elif a_result_value['labels'][0] == '属性':
                    label_map[6][a_result_value['start']] = 1
                    label_map[10][a_result_value['end']-1] = 1
                    if(len(a_result_value['text'])>2):
                        label_map[8][(a_result_value['start']+1):a_result_value['end']-1] = 1
                    start_map[a_result['id']]=a_result_value['start']

            else:
                if len(a_result['labels'])==0:
                    a_result['labels'] = ['属于']
                if a_result['labels'][0] in relations:
                    start_num = start_map[a_result['from_id']]
                    label_map[12+relations.index(a_result['labels'][0])][start_num] = 1
                    to_num = start_map[a_result['to_id']]
                    label_map[12+len(relations)+to_num][start_num] = 1
        
        prompt_ids = self.tokenizer.encode_plus(r_qb,max_length=self.max_seq_length,
                                                padding='max_length',
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')

        return  {"input_ids": prompt_ids['input_ids'].squeeze().clone().detach(),
                 "attention_mask": prompt_ids['attention_mask'].squeeze().clone().detach(), 
                 #"position_ids": torch.arange(0, self.max_seq_length).clone().detach(),
                 'text': r_qb,
                 "labels":  torch.tensor(label_map.T,dtype=torch.int).clone().detach()}

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
            datasets = load_dataset(args_data.raw_file_type, data_files={'train':'data/result.json',
                                                                        'validation':'data/result.json',
                                                                                                            })
            train_dataset = BaseDataset(args_data,tokenizer,datasets['train'],True,'do_train')
            valid_dataset = BaseDataset(args_data,tokenizer,datasets['validation'],True,'do_train')
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
            datasets = load_dataset(args_data.raw_file_type, data_files={'test':'data/test.json',
                                                                                                            })
            print(datasets)
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

        