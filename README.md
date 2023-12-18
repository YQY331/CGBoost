# A novel CGBoost deep learning algorithm for coseismic landslide susceptibility prediction

This repository represents the official implementation of the paper titled "A novel CGBoost deep learning algorithm for coseismic landslide susceptibility prediction". 

Qiyuan Yang，Xianmin Wang，Jing Yin，Aiheng Du，Aomei Zhang, Lizhe Wang, Haixiang Guo, and Dongdong Li

We present an integrated method of Crossing Graph attention network and xgBoost (CGBoost) to predict coseismic landslide susceptibility shortly after an earthquake to support efficient emergency rescue. Crossgat in CGBoost can also be applied in other applications, e.g., traffic flow prediction, recommendation systems, and classification and segmentation of point cloud.

## Repository

Clone the repository (requires git):

```bash
git clone https://github.com/YQY331/CGBoost.git
cd CGBoost
```

## Dependencies

Install denpendencies by pip:

```bash
pip install -r requirements.txt
```
## Run the model

```bash
python test_crossgat_trainedbranch.py
```

## Citation

If you use this codebase, please cite our paper:

```bibtex
@article{YANG2023101770,
      title={A novel CGBoost deep learning algorithm for coseismic landslide susceptibility prediction}, 
      author={Qiyuan Yang，Xianmin Wang，Jing Yin，Aiheng Du，Aomei Zhang, Lizhe Wang, Haixiang Guo, and Dongdong Li},
      journal = {Geoscience Frontiers},
      pages = {101770},
      year = {2023},
      issn = {1674-9871},
      doi = {https://doi.org/10.1016/j.gsf.2023.101770}
}
```

