import numpy as np
from jieba import re
from typing import Optional


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

    def get_str_all(self, objects_masks_list: list[dict]):
        # get [name, (x,y,w,h)]
        length = len(objects_masks_list)
        out = "$\\{"
        for idx in range(length):
            bb = objects_masks_list[idx]
            name = bb['name']
            x,y,w,h = bb['bbox']
            item = f"[{name}, ({x}, {y}, {w}, {h})]"
            out = out + item + (', ' if idx < length - 1 else '')
        out = out + "\\}$"
        return out

    def get_str_location(self, box_mame_list, edit_txt, size: tuple):
        # box_name_list[-1] used as a hint for GPT
        out = ""
        assert len(box_mame_list) >= 1, f'abnormal length of the box_name_list, box_name_list: {box_mame_list}'
        size_ = f"Size: ({size[0]},{size[1]})"
        list_ = "Objects: " + self.get_str_part(box_mame_list) # box_name_list[0:-1] -> box_name_list
        target_ = "Target: " + box_mame_list[-1]['name']
        edit_ = "Edit-Text: " + edit_txt
        return f'{size_}\n{list_}\n{target_}\n{edit_}'

    def get_str_part(self, objects_masks_list: list[dict]):
        # get [name, (w,h)]
        assert isinstance(objects_masks_list, list), f'type(objects_masks_list) = {type(objects_masks_list)}'
        length = len(objects_masks_list)
        out = "$\\{"
        for idx in range(length):
            item_ = objects_masks_list[idx]
            name = item_['name']
            x, y, w, h = item_['bbox']
            item = f'[{name}, ({x},{y}), ({w},{h})]'
            out = out + item + (', ' if idx < length - 1 else '')
        out = out + "\\}$"
        return out
    
    def get_str_rescale(self, old_noun, new_noun, panoptic_dict: list[dict]):
        old_noun = old_noun.strip()
        old_noun = old_noun.split(':')[-1] if ':' in old_noun else old_noun
        new_noun = new_noun.strip()
        new_noun = new_noun.split(':')[-1] if ':' in new_noun else new_noun
        Objects = self.get_str_part(panoptic_dict)
        return f'Objects: {Objects}\nOld: {old_noun}\nNew: {new_noun}'

    def get_str_add_panoptic(self, panoptic_dict: list[dict], name: str, size: tuple):
        # place not specified
        Size = f'({size[0]},{size[1]})' # (W,H)
        Objects = self.get_str_part(panoptic_dict)
        return f'Size: {Size}\nObjects: {Objects}\nTarget: {name}'

    def get_str_add_place(self, place, name, size: tuple, place_box: Optional[tuple]):
        Size = f'({size[0]},{size[1]})'
        Place = f'[{place}, ({place_box[0]},{place_box[1]}), ({place_box[2]},{place_box[3]}]'
        return f'Size: {Size}\nPlace: {Place}\nTarget: {name}'

    def __str__(self):
        out = "$\\{"
        cnt = len(self.label_dict)
        
        for (k,v) in self.label_dict.items():
            out = out + '['
            out = out + ('\'' + k + '\'')
            v = (v[0], v[1], v[2], v[3])
            out = out + ',' + v.__str__().strip() + ']'
            cnt -= 1
            if cnt >=0: out = out + ','
            else: out = out + '\\}$'
        return out


def get_replace_tuple(replace_tupple: str):
    # deal with GPT-3.5 return messages
    replace_tupple = replace_tupple.strip('(')
    replace_tupple = replace_tupple.strip(')')
    replace_tupple = replace_tupple.split(',')
    print(f'len replace_tuple = {len(replace_tupple)}')
    return (replace_tupple[0].strip(), replace_tupple[1].strip())


def get_add_tuple(add_tuple: str):
    punctuation = re.split(r'[\[(),\]:\"\']', add_tuple)
    p = [x.strip() for x in punctuation if x!='' and x!= ' ']
    assert len(p) == 3, f'p = {p}\nans = {add_tuple}'
    try:
        p[1] = int(p[1])
    except Exception as e:
        print(e)
        print('set p[1] = 3')
        p[1] = 3

    name, num, place = p[0], p[1], p[2]
    return name, num, place







