# [3DV 2025] Ctrl-Room

<h4 align="center">


Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints

<hr style="margin-top: 0; margin-bottom: 8px;">
<div align="center" style="margin-top: 0; padding-top: 0; line-height: 1;">
    <a href="https://fangchuan.github.io/ctrl-room.github.io" target="_blank" style="margin: 2px;"><img alt="Project"
    src="https://img.shields.io/badge/🌐%20Project-CtrlRoom-ffc107?color=42a5f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://arxiv.org/abs/2310.03602" target="_blank" style="margin: 2px;"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-CtrlRoom-b31b1b?logo=arxiv&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://github.com/fangchuan/Ctrl-Room" target="_blank" style="margin: 2px;"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-CtrlRoom-24292e?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://huggingface.co/Chuan99/Ctrl-Room" target="_blank" style="margin: 2px;"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CtrlRoom-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>


<p>**TL;DR**: Ctrl-Room is a generative framework to synthesize 3D Indoor Scene from text prompts. It includes two stages: the first stage create the 3D scene layout from a simple textual description of the room, the second stage generate a RGB panorama that is well-aligned to the 3D scene layout.</p>

Feel free to contact me (cfangac@connect.ust.hk) or open an issue if you have any questions or suggestions.

## 📢 News

- **2025-09-23**: Inference instructions are provided.
- **2025-09-18**: The source code and pretrained models are released.


## 📋 TODO

- [x] Release source code and pretrained models.
- [x] Release the dataset we use - [Structured3D]() with accurate bounding box annotation.
- [x] Provide detailed inference instructions for panorama generation.
- [x] Provide detailed inference instructions for panorama-reconstruction.
- [ ] Provide training instructions.


## 🔧 Installation

Tested with the following environment:
* Python 3.10
* PyTorch 2.3.1
* CUDA Version 11.8

### 1. Clone this repo
```bash
git clone https://github.com/fangchuan/Ctrl-Room.git
cd Ctrl-Room

```
### 2. Install dependencies
**To save you from the complex C++ libs dependencies between panorama_reconstruction modules, We strongly recommend using [settings/Dockerfile](settings/Dockerfile) to set up the environment.** You can build the docker image by the following command:
```bash
docker build -t ctrlroom:latest -f settings/Dockerfile .
# run the docker image
 docker run -it  --gpus all --name ctrlroom-test -v /path/to/your/data_and_code:/path/to/your/data_and_code ctrlroom:latest /bin/bash

# [deprecated] conda env setup on your local machine
conda create -n ctrlroom python=3.10 -y
conda activate ctrlroom
pip install -r settings/requirements.txt
```


## 📊 Dataset

- This project propose the [CtrlRoom Dataset](https://huggingface.co/datasets/Chuan99/Ctrl-Room-Dataset), which provides accurate 3D layout annotations for 12,615 rooms. The dataset includes 5,064 bedrooms, 3,064 living rooms, 2,289 kitchens, 698 studies, and 1,500 bathrooms. In total, it contains nearly 150,000 accurately oriented 3D bounding boxes across 25 object categories, with annotations meticulously completed by a team of three annotators over 1,200 hours.

## 🤗 Pretrained Models

All pretrained models are available at [HuggingFace🤗](https://huggingface.co/Chuan99/Ctrl-Room).

| **Model Name**                | **Fine-tined From** | **#Param.** | **Link** | **Note** |
|-------------------------------|---------------------|-------------|----------|----------|
| **bedroom_layout_gen**                   | From scratch                    | 63M            | [st3d_layout_bedroom](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_layout_bedroom.pt)         | Text-to-Bedroom-Layout         |
| **study_layout_gen**                   | From scratch                    | 63M            | [st3d_layout_study](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_layout_study.pt)         | Text-to-Study-Layout         |
| **livingroom_layout_gen**                   | From scratch                    | 63M            | [st3d_layout_livingroom](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_layout_livingroom.pt)         | Text-to-Livingroom-Layout         |
| **kitchen_layout_gen**                   | From scratch                    | 63M            | [st3d_layout_kitchen](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_layout_kitchen.pt)         | Text-to-Kitchen-Layout         |
| **bedroom_panorama_gen**                   | From [ControlNet-SD1.5](https://github.com/lllyasviel/ControlNet-v1-1-nightly?tab=readme-ov-file#controlnet-11-segmentation)                    | 1220M            | [st3d_panorama_bedroom](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_panorama_bedroom.ckpt)         | 3D Layout-to-Panorama         |
| **study_panorama_gen**                   | From [ControlNet-SD1.5](https://github.com/lllyasviel/ControlNet-v1-1-nightly?tab=readme-ov-file#controlnet-11-segmentation)                    | 1220M            | [st3d_panorama_study](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_panorama_study.ckpt)         | 3D Layout-to-Panorama         |
| **livingroom_panorama_gen**                   | From [ControlNet-SD1.5](https://github.com/lllyasviel/ControlNet-v1-1-nightly?tab=readme-ov-file#controlnet-11-segmentation)                    | 1220M            | [st3d_panorama_livingroom](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_panorama_livingroom.ckpt)         | 3D Layout-to-Panorama         |
| **kitchen_panorama_gen**                   | From [ControlNet-SD1.5](https://github.com/lllyasviel/ControlNet-v1-1-nightly?tab=readme-ov-file#controlnet-11-segmentation)                    | 1220M            | [st3d_panorama_kitchen](https://huggingface.co/Chuan99/Ctrl-Room/blob/main/st3d_panorama_kitchen.ckpt)         | 3D Layout-to-Panorama         |

### 🚀 Inference

```bash
# download above pretrained weights into the ckpts/ folder


# layout sampling of bedroom, living room, study ...
bash scripts/run_st3d_room_layout_sample.sh /path/to/your/ctrlroom_dataset /output_layout_samples

# Text-to-Layout-to-3D room meshes generation
bash scripts/run_text2bedroom_pipeline.sh /path/to/your/ctrlroom_dataset /output_samples

bash scripts/run_text2livingroom_pipeline.sh /path/to/your/ctrlroom_dataset /output_samples

bash scripts/run_text2study_pipeline.sh /path/to/your/ctrlroom_dataset /output_samples

bash scripts/run_text2kitchen_pipeline.sh /path/to/your/ctrlroom_dataset /output_samples

```
it will gives you the 3D room generations following below architecture:
* `bedroom`:
    - text_prompt.txt: the text prompt used for layout generation
    - samples_1x23x31.npz: sampled Scene Code (3D layout) in numpy format
    - `0/`: the first bedroom sample
        - scene_xxxx_xxxx_sem.png: the layout_semantic panorama image
        - scene_xxxx_xxxx_pano.png: the RGB panorama image generation aligned to the layout
        - scene_xxxx_xxxx.ply: the reconstructed 3D layout mesh in ply format
        - model.obj: the textured 3D room mesh in obj format

## 🦾 Training
### 1. Text-to-Layout

```bash
# Optional: if you find that the training script hangs over vvvvvery long time once running the train script, you might need to uninstall mip4py and install liibopenmpi-dev
pip uninstall mpi4py
sudo apt install libopenmpi-dev -y 

bash scripts/run_st3d_room_layout_train.sh  /path/to/your/ctrlroom_dataset /log_layout_training 
```

### 2. Layout-to-Panorama


## 😊 Acknowledgement
We would like to thank the authors of [DiffuScene](https://tangjiapeng.github.io/projects/DiffuScene/), [ATISS](https://github.com/nv-tlabs/ATISS), and [Sceneformer](https://github.com/cy94/sceneformer) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.


## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@article{Ctrl-Room,
    title         = {Ctrl-room: Controllable text-to-3d room meshes generation with layout constraints},
    author        = {Chuan Fang, Yuan Dong, Kunming Luo, Xiaotao Hu, Rakesh Shrestha, Ping Tan},
    journal       = {2025 International Conference on 3D Vision (3DV).},
    year          = {2025},
    eprint        = {2025: 692-701.},
    archivePrefix = {IEEE},
    primaryClass  = {cs.CV}
}

```