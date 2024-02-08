import numpy as np
import torch
import torch.nn as nn
import math
from typing import List
from torch.autograd import Variable

# for creating mask to remove padding tokens
def create_src_mask(source, padding_token_value, device):
    source_mask = (source != padding_token_value).unsqueeze(1)
    
    return source_mask.to(device)

# for creating mask to remove padding tokens and future tokens
def create_tgt_mask(target, padding_token_value, device):
    target_mask = (target != padding_token_value).unsqueeze(1)
    sequence_lenght = target.size(1)
    nopeak_mask = np.triu(np.ones((1, sequence_lenght, sequence_lenght)), k = 1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(device)
    target_mask = target_mask & nopeak_mask
    return target_mask.to(device)


def toWordLevel(instance):
    instance = ' '.join(instance.split())
    parser_list_lv1 = ['==', '!=', '&&', '||', '<=', '>=', '>>']
    parser_list_lv2 = ['!', ';', '=', '+', '-', '&', '%', '*', ':', '.', '|', '/', '(', ')', '{', '}', '[', ']', '<', '>', '\'', '\"', ',', ' ']
    
    parselv1 = []
    while len(instance) > 2:
        i = 0
        while True:
            if instance[i:i+2] in parser_list_lv1:
                if i != 0:
                    parselv1.append(instance[:i])
                parselv1.append(instance[i:i+2])
                instance = instance[i+2:]
                break
            if i == len(instance):
                parselv1.append(instance)
                instance = ''
                break
            i += 1
    parselv2 = []
    for st in parselv1:
        if st not in parser_list_lv1:
            while len(st) > 0:
                i = 0
                while True:
                    if i == len(st):
                        parselv2.append(st)
                        st = ''
                        break
                    if st[i] in parser_list_lv2:
                        if i != 0:
                            parselv2.append(st[:i])
                        parselv2.append(st[i])
                        st = st[i+1:]
                        break
                    i += 1
        else:
            parselv2.append(st)
    return parselv2

def read_corpus(file, max_len):
    
    list_of_instances = open(file, 'r').read().split('\n')
    out = []
    for i in range(len(list_of_instances)):
        instance = toWordLevel(list_of_instances[i])
        out.append(instance)
        
    print('%d instances read!                       '%len(out))
    return out

def pad_idx_tensor(idx_tensor, pad, max_len, device):
    for i in range(len(idx_tensor)):
        if len(idx_tensor[i]) < max_len:
            idx_tensor[i] += [pad] * (max_len-len(idx_tensor[i]))
        else:
            idx_tensor[i] = idx_tensor[i][:max_len]
    return torch.tensor(idx_tensor, device=device)

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_instances = [e[0] for e in examples]
        tgt_instances = [e[1] for e in examples]
        
        yield src_instances, tgt_instances

        