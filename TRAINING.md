# Training
## Requirements
- Linux with Python ≥ 3.7
- PyTorch ≥ 2.1

## Step 0: Download pretrained models and prepare training data

Download [pretrained VQGAN](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt), [pretrained image AR model](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/c2i_B_256.pt), put them in "./pretrained_models".

### Training data preparation

**Option 1: Preprocess by yourself**

For the HyperKvasir dataset, you can download from the [official website](https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-videos.zip). Once downloaded, convert the videos into 16-frame clips and further process them into 4x4 grid images.

For the SugrVisDom dataset, also can download the videos from the [official website](https://surgvisdom.grand-challenge.org/Home/), specifically using the "procine" subfolder. Similar to HyperKvasir, convert these videos into 16-frame clips and then into 4x4 grid images.

**Option 2: Use our preprocessed data**

Additionally, you can utilize our preprocessed data, which is available for download at [Hugging Face](https://huggingface.co/datasets/jeffrey423/EndoGen-Preprocessed-Videos). However, please ensure that you adhere to the original data usage guidelines. Be aware that there is a risk of data access being revoked if required by the original data owner.

You can also use your own dataset, as long as you preprocess your data similar to our data structure in {datasetname}_frames_4x4_128.

## Step 1: Pre-extract discrete codes of training data

### HyperKvasir (4x4=16 frames, 128x128 per frame)
```
bash extract_codes_c2i_512.sh
```
### HyperKvasir (8x8=64 frames, 128x128 per frame)
```
bash extract_codes_c2i_1024.sh
```
### SurgVisdom (4x4=16 frames, 128x128 per frame)
```
bash extract_codes_surgvisdom_c2i_512.sh
```
Note that you should modify the "--data-path" (where the grid images are saved) and "--code-path" (where you want to save the extracted codes) in the scripts.

## Step 2: Train AR models

### HyperKvasir (4x4=16 frames, 128x128 per frame)
```
bash train_c2i_512.sh
```
### HyperKvasir (8x8=64 frames, 128x128 per frame)
```
bash train_c2i_1024.sh
```
### SurgVisdom (4x4=16 frames, 128x128 per frame)
```
bash train_surgvisdom_c2i_512.sh
```
Note that you should modify the "--code-path" (where your extracted coded are saved previously) and "--results-dir" (where you want to save the trained AR models) in the scripts.

## Step 3: Generate videos with trained AR models

### HyperKvasir (4x4=16 frames, 128x128 per frame)
```
bash sample_c2i_512.sh
```
### HyperKvasir (8x8=64 frames, 128x128 per frame)
```
bash sample_c2i_1024.sh
```
### SurgVisdom (4x4=16 frames, 128x128 per frame)
```
bash sample_surgvisdom_c2i_512.sh
```
Note that you should modify the "--gpt-ckpt" (choose a .pt file in your trained AR model folder) and "--save-folder" (where you want to save the sampled video frames) in the scripts.

IMPORTANT: For evaluation feasibility, our code has been modified to generate the same number of video clips as the training data and may cost a while. If you just want to try simple generation, please refer to the [Simple Generation with Pretrained EndoGen](README.md#simple-generation-with-pretrained-endogen) section in the README.




### Evaluation
See ./evaluation_metrics.