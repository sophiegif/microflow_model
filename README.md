# MicroFlow 

[[arXiv](https://arxiv.org/abs/2504.13452) ] [[project page](https://jbertrand89.github.io/microflow/)]

This repository contains official code for MicroFlow: Domain-Specific Optical Flow for Ground Deformation Estimation in Seismic Events.

**[➡️ Jump directly to Inference on Real Examples](#inference-for-real-examples)**

## Installation

We recommend using a Python 3.12 virtual environment to run MicroFlow. 

### 1. Prerequisites (LAPACK)
MicroFlow depends on `prox-tv`, which requires LAPACK to be available on your system prior to installation.

**Option A: Standard Linux Install**
If you have root access (e.g., Ubuntu/Debian), you can install it easily via:
```bash
sudo apt-get update
sudo apt-get install liblapacke-dev
```
*(On CentOS/RHEL use `sudo yum install lapacke lapacke-devel`)*

**Option B: Cluster / HPC Install (Without Sudo)**
If you are on a cluster and cannot use sudo, you can compile LAPACK from scratch into your home directory:
```bash
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz
tar xf v3.12.0.tar.gz
cd lapack-3.12.0
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/lapack \
  -DBUILD_SHARED_LIBS=ON \
  -DLAPACKE=ON \
  -DCBLAS=ON
make -j$(nproc)
make install

# Export the paths so pip can find LAPACK during installation
export LD_LIBRARY_PATH=$HOME/lapack/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/lapack/lib64:$LIBRARY_PATH
export CPATH=$HOME/lapack/include:$CPATH
```

### 2. Setup Python Environment

Create a virtual environment and install the required dependencies. If you used **Option B** above, ensure you have exported the LAPACK variables before running `pip install`.

```bash
# Verify you are using Python 3.12 (e.g., `module load python/3.12.7` on clusters)
python -m venv microflow_env
source microflow_env/bin/activate

# Upgrade pip utilities
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

## Training on the semi-synthetic dataset FaultDeform

### Downloading the FaultDeform dataset
The FaultDeform dataset is a synthetic dataset that can be downloaded [here](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/G02ZXZ&version=1.0).
You can download it, it will follow the following file structure:

```
FaultDeform_root_dir
├── train
│   │── 000000_0_sample.npy.npz
│   │── 000000_1_sample.npy.npz
│   │── 000000_2_sample.npy.npz
│   │── 000000_3_sample.npy.npz
│   └── ...
├── val
│   │── 000000_0_sample.npy.npz
│   │── 000000_1_sample.npy.npz
│   │── 000000_2_sample.npy.npz
│   │── 000000_3_sample.npy.npz
│   └── ...
└── test
    │── 000000_0_sample.npy.npz
    │── 000000_1_sample.npy.npz
    │── 000000_2_sample.npy.npz
    │── 000000_3_sample.npy.npz
    └── ...
```

### Training Microflow

To train MicroFlow, run
```
python -u train_fault_deform.py \
--train_config_name irseparated_geoflownet_intermediatel1_noreg \
--checkpoints_dir <your_checkpoint_dir> \
--offline_dir <your_offline_dir> \
--save_offline \
--dataset_dir <FaultDeform_root_dir> \
--split_dir <your_directory_containing_the_split_files_for_fault_deform> \
--seed 1 \
--device cuda
```
using the pre-saved config located in `data/configs/train_fault/deform`
and setting 
- `checkpoint_dir`: the directory for saving the checkpoints
- `offline_dir`: the directory for saving the wandb offline logs if you setup `--save_offline`
- `dataset_dir`: the root directory of Fault Deform
- `split_dir`: the directory containing the split files
- `device`: the hardware device to run training on (`cuda` or `cpu`)

### Pre-saved configs
Pre-saved configs for the pretrained models can be found in `data/configs/train_fault/deform`.

To create a new config, use the code in `src/configs/save_train_config.py`.
Fow example, to create the `irseparated_geoflownet_intermediatel1_noreg` config, run:
```
python src/configs/save_train_config.py \
--config_filename data/configs/train_fault_deform/irseparated_geoflownet_intermediatel1_noreg.yaml \
--amp \
--model_name irseparated_GeoFlowNet \
--dataset_dir <your_directory_containing_fault_deform> \
--split_dir <your_directory_containing_the_split_files_for_fault_deform> 
```

### Pre-trained models and real data crops

The temporary access to the models and real data 1024x1024 crops is here: https://drive.google.com/drive/folders/1PB2yfOJH3e9VcFZySrmymd-4xDj2NzhP?usp=sharing

You can then add the weights in the `pretrained_models/'`folder.

[old versions: 
You can find the models trained for reproducing the paper on [huggingface](https://huggingface.co/zjuzju/microflow_models). 
You can either download the full repository, by running the following python code
```
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="zjuzju/microflow_models")
```
or download each model separately using wget
```
wget https://huggingface.co/zjuzju/microflow_models/resolve/main/irseparated_geoflownet_intermediatel1/irseparated_GeoFlowNet_intermediatel1_0.8_sf012_e40.pt
```
]

## Inference of FaultDeform

Run the following command
```
python inference_fault_deform.py \
--train_config_name irseparated_geoflownet_intermediatel1_noreg \
--pretrained_model_filename <your_pretrained_model> \
--metric_filename <your_metric_filename> \
--save_metrics \
--dataset_name faultdeform \
--dataset_dir <FaultDeform_root_dir> \
--split_dir <your_directory_containing_the_split_files> \
--split_scaling_factors 1 \
--device cuda
```

and specify
- `pretrained_model_filename`: the path for your model (.pt)
- `metric_filename`: the path where to save your results
- `dataset_dir`: the root directory of Fault Deform
- `split_dir`: the directory containing the split files
- `split_scaling_factors`: either 0 (very small displacements), 1 (small displacements) or 2 (large displacements)
- `device`: the hardware device to run inference on (`cuda` or `cpu`)

Note that your config must be located in `data/configs/train_fault_deform`.

## Inference for real examples

### Real Examples
You can evaluate our model on any pair of real-world examples, following the file structure
```
Real_example_root_dir
├── first_example_dir
│   │── <first_example_template>_pre.tif
│   └── <first_example_template>_post.tif
├── second_example_dir
│   │── <second_example_template>_pre.tif
│   └── <second_example_template>_post.tif
└── ...
```

### Inference

Depending on your image size and the quality required, you can configure the sliding window via the `--window_size`, `--stride`, and `--offset` parameters.

Additionally, choose your desired architecture via the `--config_name`:
- `--config_name microflow` : Uses the full model **with regularization** (smoother, heavily regularized output).
- `--config_name iterative_only` : Uses the model **without regularization** (faster, but potentially noisier raw optical flow).

Hardware acceleration can be controlled via the `--device` flag (`--device cuda` for GPU, or `--device cpu` to use system RAM and CPU).

#### Option 1: Fast Computation (Small Patches)
If your image is small (e.g., fits securely within the `window_size` such as 1024x1024) or you just want a very fast initial result without blending overlapping patches, use `stride=0` and `offset=0`:

```bash
python inference_real_examples.py \
  --config_name iterative_only \
  --pretrained_model_filename pretrained_models/microflow_seed10.pt \
  --dataset_dir data/example_real_data/ \
  --save_dir data/example_real_data/results \
  --window_size 1028 \
  --stride 0 \
  --offset 0 \
  --batch-size 1 \
  --device cuda
```

#### Option 2: Clean Solution (Large Images)
For large satellite images, processing in overlapping chunks with edge trimming yields the cleanest, artifact-free mosaicked results. We recommend a `window_size` of effectively 1024 (passed as 1028 for padding constraints), an overlap (`stride`) of 512, and an `offset` of 128 to remove boundary edge effects from each patch before blending.

```bash
python inference_real_examples.py \
  --config_name microflow \
  --pretrained_model_filename pretrained_models/microflow_seed10.pt \
  --dataset_dir data/example_real_data/ \
  --save_dir data/example_real_data/results/ \
  --window_size 1028 \
  --stride 512 \
  --offset 128 \
  --batch-size 1 \
  --device cuda
```

### Parameter Explanations
- `config_name`: `microflow` (regularized) or `iterative_only` (unregularized). You can also use the  `searaft` finetuned model.
- `pretrained_model_filename`: Path to the `.pt` weights. (microflow_seed10.pt or searaft_s10.pt)
- `dataset_dir`: Contains the `_pre.tif` and `_post.tif` files for your real example.
- `save_dir`: Directory to save the resulting optical flow (`.tif`) estimates.
- `window_size`: The patch size fed into the model (1028 recommended).
- `stride`: Step size between patches (use `0` for no overlap, `512` for 50% overlap).
- `offset`: Number of boundary pixels to trim off each patch before blending (e.g. `128`) to prevent edge artifacts.
- `device`: Set to `cpu` or `cuda`.

## Citation
Coming soon


