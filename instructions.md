## Setup
This is done in the login node of CLIP

Set up environment only once:

- Install Anaconda
- load the up-to-date compile toochain: ml build-env/f2022; ml foss/2023a
- Make new environment: conda create -n weaver python=3.12
- Install pytorch (CPU platform): pip3 install torch torchvision torchaudio
- Install weaver-core: pip3 install weaver-core
- Install cmake: pip3 install cmake
- Install onnx: pip3 install onnx (to have onnxruntime, python version needs to be lower than 3.10)
- Install pyarrow: conda install -c conda-forge pyarrow
- Install ParT: git clone git@github.com:Ang-Li-95/particle_transformer.git

## Workflow

First load the environment: `conda activate weaver` and go to the work directory: `cd /path/to/particle_transformer`

### 1. Prepare the input data ntuple

There are two options to get the ntuple:
- Start from MiniAOD and [run MLTree](https://github.com/HephyAnalysisSW/SoftDisplacedVertices/blob/main/ML/test/submit_jobs.py). Make sure to use updated JSON file `/users/ang.li/public/SoftDV/CMSSW_10_6_30/src/SoftDisplacedVertices/Samples/json/CustomMiniAOD_v3_MLTraining.json`.
- Simply use NanoAOD(JSON file available in `/users/ang.li/public/SoftDV/CMSSW_10_6_30/src/SoftDisplacedVertices/Samples/json/CustomNanoAOD_v3_MLTraining.json`)

### 2. Convert the ntuple to `parquet` format using the needed variables.

[A script](https://github.com/Ang-Li-95/particle_transformer/blob/main/divide_samples.py) could help in doing this, modify the input and tree names to make it work.

### 3. Prepare the data config

General description of data config can be found [here](https://github.com/hqucms/weaver-core?tab=readme-ov-file#data-configuration-file). One example of the config is shown [here](https://github.com/Ang-Li-95/particle_transformer/blob/main/data/LLP/LLP_vtx.yaml). Make sure variables in the config file is available in the `parquet` file.

### 4. Train the model

Run `./train_LLP.sh ParT vtx /path/to/data`

