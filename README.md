# WaveMo: Learning Wavefront Modulations to See Through Scattering
​
Code for the CVPR 2024 paper "WaveMo: Learning Wavefront Modulations to See Through Scattering".
<br>[**Project Page**](https://wavemo-2024.github.io/) | [**PDF**](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_WaveMo_Learning_Wavefront_Modulations_to_See_Through_Scattering_CVPR_2024_paper.pdf) | [**arXiv**](https://arxiv.org/abs/2404.07985) | [**YouTube**](https://www.youtube.com/watch?v=2iP-0nzV6TY) <br>

## Installation
We test our code with Torch 2.3.1 and CUDA 12.4.1 on a NVIDIA RTX A6000 GPU. Please follow these steps to set up the environment:
``` 
conda create -n wavemo python=3.8.11
conda activate wavemo
pip install -r requirements.txt
``` 


## Dataset
We use [Places365](http://places2.csail.mit.edu/index.html) as our training dataset. Please acquire it from http://places2.csail.mit.edu/download.html. You can also use other image datasets for training.


## Learn Wavefront Modulations

Please run the following command to learn wavefront modulation patterns. The reconstruction results on the test set, network checkpoints, and learned modulations will all be saved under OUTPUT_DIRECTORY. If you come across GPU Memory issues, please decrease the batch size.

``` 
python main.py --training_data_dir DATASET_DIRECTORY --batch_size 32
```

If you have a [Weights & Biases](https://wandb.ai/home) account and want to log your training results onto it, please run 

``` 
python main.py --training_data_dir DATASET_DIRECTORY --batch_size 32 --use_wandb
```


## Apply Learned Wavefront Modulations to Real Experiments
You need to first set up the optical system as described in Section 6 of our paper. Then you can load the learned modualtions onto the spatial light modulator to modulate the measurements. To reconstruct the target, we recommend to use the unsupervised iterative approach proposed by [Feng et al.](https://www.science.org/doi/10.1126/sciadv.adg4671), the code of which is available at https://github.com/Intelligent-Sensing/NeuWS.


## Contact
For general questions of this project, please raise an issue here or contact Mingyang Xie at mingyang@umd.edu. For questions related to setting up the optical system, please contact Haiyun Guo at hg39@rice.edu.