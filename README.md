# CLIPN (ICCV 2023)
# This is the official implementation for our paper "CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No" (ICCV 2023)


## 1. News
- Our paper is accepted by ICCV 2023. Thanks to the valuable suggestions from the reviewers of CVPR 2023 and ICCV 2023, our paper has been significantly improved, allowing it to be published at ICCV 2023.


## 2. Install
- Main python libraries of our experimental environment are shown in [requirements.txt](./requirements.txt). You can install CLIPN following:
```shell
git clone https://github.com/xmed-lab/CLIPN.git
cd CLIPN
conda create -n CLIPN
conda activate CLIPN
pip install -r ./requirement.txt
```

- We also provide the original conda env of our experiments, you can download here and install it:
```shell
git clone https://github.com/xmed-lab/CLIPN.git
cd CLIPN
conda create -n CLIPN --clone dir_of_your_download_env
conda activate CLIPN
```

## 3. Prepare Dataset
- Pre-training Dataset, CC3M.
To download CC3M dataset as webdataset, please follow [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md).

When you have downloaded CC3M, please re-write your data root into [./src/run.sh](./src/run.sh).

- OOD detection datasets.
   - ID dataset, ImageNet-1K: The ImageNet-1k dataset (ILSVRC-2012) can be downloaded [here](https://image-net.org/challenges/LSVRC/2012/index.php#).
   - OOD dataset, iNaturalist, SUN, Places, and Texture. Please follow instruction from the these two repositories [MOS](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) and [MCM](https://github.com/deeplearning-wisc/MCM) to download the subsampled datasets where semantically overlapped classes with ImageNet-1k are removed.

When you have downloaded the above datasets, please re-write your data root into [./src/tuning_util.py](./src/tuning_util.py).

## 4. Pre-Train and Evaluate CLIPN

- Pre-train CLIPN on CC3M.
   - The model of CLIPN is defined in [./src/open_clip/model.py](./src/open_clip/model.py)
   - The function of loading parameters of CLIP is defined in [./src/open_clip/factory.py](./src/open_clip/factory.py).
   - The loss functions are defined in [./src/open_clip/loss.py](./src/open_clip/loss.py).
   - You can pre-train CLIPN on ViT-B-32 and ViT-B-16 by:
```shell
cd ./src
sh run.sh
```

- Zero-Shot Evaluate CLIPN on ImageNet-1K.
   - Metrics and pipeline are defined in [./src/zero_shot_infer.py](./src/zero_shot_infer.py).
   - Dataset details are defined in [./src/tuning_util.py](./src/tuning_util.py).
   - Inference models are defined in [./src/classification.py](./src/classification.py), including converting the text encoders into classifiers.

   - You can evaluate CLIPN by:
```shell
python3 zero_shot_infer.py
```


## 5. Reproduced Results

***To ensure the reproducibility of the results, we conducted three repeated experiments under each configuration. The following will exhibit the most recent reproduced results achieved before open-sourcing.***

- ImageNet-LT
<table>
    <tr align="center">
        <td rowspan="2">Methods</td>
        <td rowspan="2">Repeat</td>
        <td colspan="2">iNaturalist</td>
        <td colspan="2">SUN</td>
        <td colspan="2">Textures</td>
        <td colspan="2">Places</td>
        <td colspan="2">Avg</td>
        <td rowspan="2">Model/log</td>
    </tr>
    <tr align="center">
       <td>AUROC</td>
       <td>FPR95</td>
       <td>AUROC</td>
       <td>FPR95</td>
       <td>AUROC</td>
       <td>FPR95</td>
       <td>AUROC</td>
       <td>FPR95</td>
       <td>AUROC</td>
       <td>FPR95</td>
    </tr>
    <tr align="center">
       <td colspan="13">ViT-B-16</td>
    </tr>
    <tr align="center">
       <td rowspan="4">CLIPN-CTW</td>
       <td>1</td>
       <td>93.12</td>
       <td>26.31</td>
       <td>88.46</td>
       <td>37.67</td>
       <td>79.17</td>
       <td>57.14</td>
       <td>86.14</td>
       <td>43.33</td>
       <td>_</td>
       <td>_</td>
       <td><a href="https://drive.google.com/drive/folders/1CRIKr0vwrvK4Mc63zfhg2o8cbEGct4oF?usp=sharing">here</a></td>
    </tr>
    <tr align="center">
       <td>2</td>
       <td>93.48</td>
       <td>21.06</td>
       <td>89.79</td>
       <td>30.31</td>
       <td>83.31</td>
       <td>46.44</td>
       <td>88.21</td>
       <td>33..85</td>
       <td>_</td>
       <td>_</td>
       <td><a href="https://drive.google.com/drive/folders/1eNaaPaRWz0La8_qQliX30A4I7Y44yDMY?usp=sharing">here</a></td>
    </tr>
    <tr align="center">
       <td>3</td>
       <td>91.79</td>
       <td>25.84</td>
       <td>89.76</td>
       <td>31.30</td>
       <td>76.76</td>
       <td>59.25</td>
       <td>87.66</td>
       <td>36.58</td>
       <td>_</td>
       <td>_</td>
       <td><a href="https://drive.google.com/drive/folders/1qF4Pm1JSL3P0H4losPSmvldubFj91dew?usp=sharing">here</a></td>
    </tr>
    <tr align="center">
       <td>Avg</td>
       <td>92.80</td>
       <td>24.41</td>
       <td>89.34</td>
       <td>33.09</td>
       <td>79.75</td>
       <td>54.28</td>
       <td>87.34</td>
       <td>37.92</td>
       <td>87.31</td>
       <td>37.42</td>
       <td>_</td>
    </tr>
    <tr align="center">
       <td rowspan="4">CLIPN-ATD</td>
       <td>1</td>
       <td>95.65</td>
       <td>21.73</td>
       <td>93.22</td>
       <td>29.51</td>
       <td>90.35</td>
       <td>42.89</td>
       <td>91.25</td>
       <td>36.98</td>
       <td>_</td>
       <td>_</td>
       <td><a href="https://drive.google.com/drive/folders/1CRIKr0vwrvK4Mc63zfhg2o8cbEGct4oF?usp=sharing">here</a></td>
    </tr>
    <tr align="center">
       <td>2</td>
       <td>96.67</td>
       <td>16.71</td>
       <td>94.77</td>
       <td>23.41</td>
       <td>92.46</td>
       <td>34.73</td>
       <td>93.39</td>
       <td>29.24</td>
       <td>_</td>
       <td>_</td>
       <td><a href="https://drive.google.com/drive/folders/1eNaaPaRWz0La8_qQliX30A4I7Y44yDMY?usp=sharing">here</a></td>
    </tr>
    <tr align="center">
       <td>3</td>
       <td>96.29</td>
       <td>18.90</td>
       <td>94.55</td>
       <td>24.15</td>
       <td>89.61</td>
       <td>45.12</td>
       <td>93.23</td>
       <td>30.11</td>
       <td>_</td>
       <td>_</td>
       <td><a href="https://drive.google.com/drive/folders/1qF4Pm1JSL3P0H4losPSmvldubFj91dew?usp=sharing">here</a></td>
    </tr>
    <tr align="center">
       <td>Avg</td>
       <td>96.20</td>
       <td>19.11</td>
       <td>94.18</td>
       <td>25.69</td>
       <td>90.81</td>
       <td>40.91</td>
       <td>92.62</td>
       <td>32.11</td>
       <td>93.45</td>
       <td>29.46</td>
       <td>_</td>
    </tr>
    
</table>

<font color='red'> The performance in this table is better than our paper </font>, because that we add an average learnable "no" prompt (see ***Line 600-616*** in [./src/open_clip/model.py](./src/open_clip/model.py)).



## 6. Citation

If you find our paper helps you, please kindly consider citing our paper in your publications.
```bash
@misc{wang2023clipn,
      title={CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No}, 
      author={Hualiang Wang and Yi Li and Huifeng Yao and Xiaomeng Li},
      year={2023},
      eprint={2308.12213},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 7. Acknowledge
We sincerely appreciate these three highly valuable repositories [open_clip](https://github.com/mlfoundations/open_clip), [MOS](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) and [MCM](https://github.com/deeplearning-wisc/MCM).

