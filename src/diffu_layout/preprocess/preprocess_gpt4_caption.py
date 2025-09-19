import os
import sys
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import argparse

from improved_diffusion.clip_util import FrozenCLIPEmbedder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess gpt4 caption for each room in structured3d")
    parser.add_argument("--dataset_path",
                        default="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/",
                        help="processed dataset path",
                        metavar="DIR")
    parser.add_argument("--room_types",
                        default=["bedroom", "living_room", "kitchen"],
                        help="room types",
                        metavar="LIST")
    parser.add_argument("--gpt_caption_path",
                        default="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240227_gpt_vision_caption_kitchen/",
                        help="gpt caption path",
                        metavar="DIR")
    args = parser.parse_args()
    room_type_lst = args.room_types
    splits = ['train', 'test']
    
    dataset_folder_lst = [os.path.join(args.dataset_path, split, room_type) for room_type in room_type_lst for split in splits]
    print(f"dataset_folder_lst: {dataset_folder_lst}")
    
    clip_model = FrozenCLIPEmbedder(device='cuda')

    captioned_room_lst = [f for f in os.listdir(args.gpt_caption_path) if f.endswith('.json') and f.startswith('scene_')]
    for data_folder in dataset_folder_lst:

        room_description_folder = os.path.join(data_folder, 'text_desc')
        room_desc_embedding_folder = os.path.join(data_folder, 'text_desc_emb')
        for room_id in captioned_room_lst:
            if not os.path.exists(os.path.join(room_description_folder, room_id[:-5] + '.txt')) \
                or not os.path.exists(os.path.join(room_desc_embedding_folder, room_id[:-5] + '.npy')):
                    continue
                
            captioned_filepath = os.path.join(args.gpt_caption_path, room_id)
            
            with open(captioned_filepath, 'r') as f:
                gpt_caption_dict = json.load(f)

            text_desc = gpt_caption_dict['choices'][0]["message"]["content"]
            # print(text_desc.split('\n'))
            # only use the first sentence
            text_desc = text_desc.split('\n')[0]
            
            # begin_pos = text_desc.find('of') + 2
            # cut_text_desc = text_desc[begin_pos:]
            cut_text_desc = text_desc
            cut_text_desc = cut_text_desc.strip()
            # set first letter to upper case
            cut_text_desc = cut_text_desc[0].upper() + cut_text_desc[1:]
            print(f"room {room_id[:-5]} cut_text_desc:{cut_text_desc}")
            
            new_text_desc_filepath = os.path.join(room_description_folder, room_id[:-5] + '_gpt4.txt')
            with open(new_text_desc_filepath, 'w') as f:
                f.write(cut_text_desc)
            
            new_text_desc_embedding_filepath = os.path.join(room_desc_embedding_folder, room_id[:-5] + '_gpt4.npy')
            desc_embedding = clip_model.get_text_embeds(cut_text_desc).cpu().numpy()
            np.save(new_text_desc_embedding_filepath, desc_embedding)
    
    