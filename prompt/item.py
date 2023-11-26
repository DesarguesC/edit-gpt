import numpy as np


class Label():
    def __init__(self):
        self.label_dict = {}
        self.idx_list = []   # index in masks list gaining from SAM
    
    def add(self, box: tuple, name: str = None, idx: int = None):
        # idx: index in original mask list
        assert name != None and idx != None and len(box)==4, f'name = {name}, idx = {idx}, box = {box}'
        if name not in self.label_dict.keys(): self.label_dict[name] = box
        else: pass  # ?
        self.idx_list.append(idx)
        
    def __str__(self):
        out = "{"
        cnt = len(self.label_dict)
        
        for (k,v) in self.label_dict.items():
            out = out + '['
            out = out + ('\'' + k + '\'')
            v = (v[0], v[1], v[2], v[3])
            out = out + ',' + v.__str__().strip() + ']'
            cnt -= 1
            if cnt >=0: out = out + ','
            else: out = out + '}'
        return out


def get_replace_tuple(replace_tupple: str):
    # deal with GPT-3.5 return messages
    replace_tupple = replace_tupple.strip('(')
    replace_tupple = replace_tupple.strip(')')
    replace_tupple = replace_tupple.split(',')
    print(f'len replace_tuple = {len(replace_tupple)}')
    return (replace_tupple[0], replace_tupple[1])

