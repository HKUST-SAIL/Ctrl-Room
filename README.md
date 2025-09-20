# Ctrl-Room
This repo contains code for Ctrl-Room.

This repository contains the official implementation of the paper: [Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints](https://arxiv.org/abs/2310.03602)
Ctrl-Room is a generative framework to synthesize 3D Indoor Scene from text prompts. It includes two stages: the first stage create the 3D scene layout from a simple textual description of the room, the second stage generate a RGB panorama that is well-aligned to the 3D scene layout.

Feel free to contact me (cfangac@connect.ust.hk) or open an issue if you have any questions or suggestions.

## 📢 News

- **2025-09-18**: Inference instructions are provided.
- **2025-03-29**: The source code and pretrained models are released.


## 📋 TODO

- [x] Release source code and pretrained models.
- [x] Release the dataset we use - [Structured3D]() with accurate bounding box annotation.
- [x] Provide detailed inference instructions for 3D scene generation.
- [ ] Provide training instructions.
- [ ] Implement a Gradio demo at HuggingFace🤗 Space.


## 🔧 Installation

source code:
```bash
git clone https://github.com/fangchuan/Ctrl-Room.git
cd Ctrl-Room && git checkout develop
```

docker image:
```bash
docker pull registry.cn-beijing.aliyuncs.com/scenelrm/roomverse:v3
docker run -it --gpus all --rm -v /your/path/to/data:/path/to/data registry.cn-beijing.aliyuncs.com/scenelrm/roomverse:v2 /bin/bash
```

## 📊 Dataset

- We use [CtrlRoom Dataset](oss://mri-ai-training/data/dataset/qunhe/PanoRoom/processed_data_8000.tar.gz) with about 8.2K 3D room_layout and 820K rendered images (8.2K x 100 views, including RGB, normal, depth maps and semantic map) for MVD training. 

## 🚀 Usage


## 😊 Acknowledgement
We would like to thank the authors of [SceneCraft](https://github.com/OrangeSodahub/SceneCraft), [Set-the-Scene](https://github.com/DanaCohen95/Set-the-Scene), and [Wonder3D](https://www.xxlong.site/Wonder3D) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.


## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{Ctrl-Room,
  title={xxxx},
  author={xxxx and xxxx and xxxx and xxxx and xxxx},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```