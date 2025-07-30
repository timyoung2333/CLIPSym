# CLIPSym: Delving into Symmetry Detection with CLIP
This code contains instruction on how to reproduce our results in the paper.

## Requirements
Run the following commands to create a virtual environment, activate it and install the required packages.
```
conda create -n clipsym python=3.8.18
conda activate clipsym
pip install -r requirements.txt
```

## Datasets
DENDI dataset can be downloaded from [DENDI](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/ES2ftVVmTc5Du78EBgfTGy8BwygV_HRa5nWciYeq3cTvoQ?e=y9ETja).

SDRW and LDRS datasets can be downloaded from [PMC](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/EQdRWpc9HiRDqgdQohA3X-oBuoeUS6d8U24dRykhsL1vnw?e=eQ2vaN) (password: ldrs2021).

After downloaded, please unzip dataset zip files under the `sym_datasets` folder. Note, `sym_datasets` contains a csv file `objects.csv`, which contains frequent object classes in the DENDI dataset.

## 3. Experiments

### Usage
```python
python train.py --dataset <DATASET> --method <METHOD> [OTHER PARAMETERS]
```
The script for training CLIPSym is
```
python train.py --dataset dendi --method clipsym -eq_up
```
### Arguments
```
'--dataset': 'dendi' or 'pmc'
'--method': 'clipsym', 'equisym', 'pmc'
'-eq_up': should be used together with method 'clipsym'. When specified, the upsampler is G-equivariant.
```

Please check config.py for other hyperparameters. If they are not set, the code will be using the default parameters for reproducing the results reported in the paper.

You may refer https://github.com/ahyunSeo/EquiSym and https://github.com/ahyunSeo/PMCNet for more details about EquiSym and PMCNet.