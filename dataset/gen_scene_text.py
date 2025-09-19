import random
import torch
from num2words import num2words
from nltk.tokenize import word_tokenize
from .text_utils import compute_rel, get_article
from collections import Counter, defaultdict

from typing import List, Dict, Tuple, Any, Optional, Union, Callable

import numpy as np



def dict_bbox_to_vec(dict_box):
    '''
    input: {'min': [1,2,3], 'max': [4,5,6]}
    output: [1,2,3,4,5,6]
    '''
    return dict_box['min'] + dict_box['max']


def clean_obj_name(name):
    return name.replace('_', ' ')


def ordered_box_with_size(room_type: str, object_dicts: Dict):

    object_dict_cp = {}
    object_dict_cp['objects'] = []
    # discard 'door' object
    for obj in object_dicts['objects']:
        # here we skip door and curtain , since door is default in the room, and curtain is too large in Structured3D
        if obj['class'] not in ['door', 'curtain']:
            object_dict_cp['objects'].append(obj)

    if len(object_dict_cp['objects']) == 0:
        object_dict_cp['objects'] = object_dicts['objects']
    bbox_size_lst = np.array([np.array(bbox['size']) for bbox in object_dict_cp['objects']])

    ordering = np.lexsort(np.hstack([bbox_size_lst]).T)

    ordered_bboxes = [object_dict_cp['objects'][i] for i in ordering[::-1]]
    return ordered_bboxes


def add_relation(objct_dict: Dict):
    '''
        Add relations to sample['relations']
    '''
    relations = []
    num_objs = len(objct_dict['objects'])

    for ndx in range(num_objs):
        this_box = objct_dict['objects'][ndx]
        this_box_trans = np.array(this_box['center'])
        this_box_sizes = (np.array(this_box['size']) * 0.5)
        this_box = {'min': list(this_box_trans - this_box_sizes), 'max': list(this_box_trans + this_box_sizes)}

        # only backward relations
        choices = [other for other in range(num_objs) if other < ndx]
        for other_ndx in choices:
            prev_box = objct_dict['objects'][other_ndx]
            prev_box_trans = np.array(prev_box['center'])
            prev_box_sizes = (np.array(prev_box['size']) * 0.5)
            prev_box = {'min': list(prev_box_trans - prev_box_sizes), 'max': list(prev_box_trans + prev_box_sizes)}
            box1 = dict_bbox_to_vec(this_box)
            box2 = dict_bbox_to_vec(prev_box)

            relation_str, distance = compute_rel(box1, box2)
            if relation_str is not None:
                relation = (ndx, relation_str, other_ndx, distance)
                # print('box: {}, center: {} is {} with {} center {}'.format(objct_dict['objects'][ndx]['class'],
                #                                                            objct_dict['objects'][ndx]['center'],
                #                                                            relation_str,
                #                                                            objct_dict['objects'][other_ndx]['class'],
                #                                                            objct_dict['objects'][other_ndx]['center']))

                relations.append(relation)

    objct_dict['relations'] = relations

    return objct_dict


def add_description(room_type: str, object_dict: Dict, eval=False):
    '''
        Add text descriptions to each scene
        sample['description'] = str is a sentence
        eg: 'The room contains a bed, a table and a chair. The chair is next to the window'
    '''
    sentences = []
    # clean object names once
    obj_names = list(map(clean_obj_name, [obj['class'] for obj in object_dict['objects']]))
    # skip the first object in Structured3D dataset, which is always the curtain
    # obj_names = obj_names[1:]
    # objects that can be referred to
    refs = []
    # TODO: handle commas, use "and"
    # TODO: don't repeat, get counts and pluralize
    # describe the first 2 or 3 objects
    if eval:
        first_n = 3
    else:
        first_n = random.choice([2, 3])
    # first_n = len(obj_names)
    first_n_names = obj_names[:first_n]
    first_n_counts = Counter(first_n_names)

    s = 'The room has '
    for ndx, name in enumerate(sorted(set(first_n_names), key=first_n_names.index)):
        if ndx == len(set(first_n_names)) - 1 and len(set(first_n_names)) >= 2:
            s += "and "
        if first_n_counts[name] > 1:
            s += f'{num2words(first_n_counts[name])} {name}s '
        else:
            s += f'{get_article(name)} {name} '
        if ndx == len(set(first_n_names)) - 1:
            s += ". "
        if ndx < len(set(first_n_names)) - 2:
            s += ', '
    sentences.append(s)
    refs = set(range(first_n))

    # for each object, the "position" of that object within its class
    # eg: sofa table table sofa
    #   -> 1    1    2      2
    # use this to get "first", "second"

    seen_counts = defaultdict(int)
    in_cls_pos = [0 for _ in obj_names]
    for ndx, name in enumerate(first_n_names):
        seen_counts[name] += 1
        in_cls_pos[ndx] = seen_counts[name]

    for ndx in range(1, len(obj_names)):
        # higher prob of describing the 2nd object
        prob_thresh = 0.01

        if eval:
            random_num = 1.0
        else:
            random_num = random.random()
        if random_num > prob_thresh:
            # possible backward references for this object
            possible_relations = [r for r in object_dict['relations'] \
                                    if r[0] == ndx \
                                    and r[2] in refs \
                                    and r[3] < 1.5]
            if len(possible_relations) == 0:
                continue
            # now future objects can refer to this object
            refs.add(ndx)

            # if we haven't seen this object already
            if in_cls_pos[ndx] == 0:
                # update the number of objects of this class which have been seen
                seen_counts[obj_names[ndx]] += 1
                # update the in class position of this object = first, second ..
                in_cls_pos[ndx] = seen_counts[obj_names[ndx]]

            # pick any one
            if eval:
                (n1, rel, n2, dist) = possible_relations[0]
            else:
                (n1, rel, n2, dist) = random.choice(possible_relations)
            o1 = obj_names[n1]
            o2 = obj_names[n2]

            # prepend "second", "third" for repeated objects
            if seen_counts[o1] > 1:
                o1 = f'{num2words(in_cls_pos[n1], ordinal=True)} {o1}'
            if seen_counts[o2] > 1:
                o2 = f'{num2words(in_cls_pos[n2], ordinal=True)} {o2}'

            # dont relate objects of the same kind
            if o1 == o2:
                continue

            a1 = get_article(o1)

            if 'touching' in rel:
                if ndx in (1, 2):
                    s = F'The {o1} is next to the {o2}'
                else:
                    s = F'There is {a1} {o1} next to the {o2}'
            elif rel in ('left of', 'right of'):
                if ndx in (1, 2):
                    s = f'The {o1} is to the {rel} the {o2}'
                else:
                    s = f'There is {a1} {o1} to the {rel} the {o2}'
            elif rel in ('surrounding', 'inside', 'behind', 'in front of', 'on', 'above'):
                if ndx in (1, 2):
                    s = F'The {o1} is {rel} the {o2}'
                else:
                    s = F'There is {a1} {o1} {rel} the {o2}'
            s += ' . '
            sentences.append(s)

    # set back into the sample
    object_dict['description'] = sentences

    # delete sample['relations']
    del object_dict['relations']
    return object_dict


def add_glove_embeddings(glove_mode, object_dicts: Dict, max_sentences=3, max_token_length=77):

    sentence = object_dicts['description']
    # tokens = list(word_tokenize(sentence))
    # # pad to maximum length
    # tokens += ['<pad>'] * (max_token_length - len(tokens))

    # embed words
    # object_dicts['desc_emb'] = torch.cat([glove_mode[token].unsqueeze(0) for token in tokens]).numpy()
    object_dicts['desc_emb'] = glove_mode.get_text_embeds(sentence).cpu().numpy()
    # print(f'desc_emb.shape: {object_dicts["desc_emb"].shape}')
    return object_dicts


def get_scene_description(room_type: str,
                          wall_dict: Dict,
                          object_dict: Dict,
                          eval: bool = False,
                          glove_model=None,
                          max_sentences=3,
                          max_token_length=77,
                          use_object_ordering: bool = True):
    '''
        Returns a string description of a scene
    '''
    # generate a description of walls
    wall_num = len(wall_dict['walls'])
    scene_desc = f'The {clean_obj_name(room_type)} has {num2words(wall_num)} walls. '

    if use_object_ordering:
        # sort object name by frequency
        object_dict['objects'] = ordered_box_with_size(room_type=room_type, object_dicts=object_dict)

    # generate a description of objects
    object_dict = add_relation(object_dict)
    object_dict = add_description(room_type, object_dict, eval=eval)
    sentence = ''.join(object_dict['description'][:max_sentences])
    scene_desc += sentence
    object_dict['description'] = scene_desc

    if glove_model is not None:
        object_dict = add_glove_embeddings(glove_model,
                                           object_dict,
                                           max_sentences=max_sentences,
                                           max_token_length=max_token_length)
    del object_dict['description']

    text_emb = None
    if 'desc_emb' in object_dict:
        text_emb = object_dict['desc_emb']
        del object_dict['desc_emb']
    return scene_desc, text_emb
