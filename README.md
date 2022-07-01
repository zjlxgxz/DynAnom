# Codebase for _[Subset Node Anomaly Tracking over Large Dynamic Graphs](https://arxiv.org/abs/2205.09786)_ @KDD-2022
## Authors: _[Xingzhi Guo](https://zjlxgxz.github.io/zjlxgxz/), [Baojian Zhou](https://baojian.github.io) and [Steven Skiena](https://www3.cs.stonybrook.edu/~skiena/)_

## 1. Download data:
Download the compiled Person-Event Graph, and unzip them into the folder toy-data
- Person-Event Graph Link:
```
 https://drive.google.com/drive/folders/1nu-2Lx80WGD9I7cjXZ1H1SKauTr_UIe7?usp=sharing
```
## 2. Set up env

```
sh ./recipe/setup.sh
```
##### Notes  1. Due to weiredness of Numba, a specific python version (3.7) is needed: #SEE: https://github.com/numba/numba/issues/5156
##### Notes  2. That is a weighted version of DynamicPPE  


## 2. Run the method (all experiments)
```
sh ./recipe/run.sh
```

## 3. Output figure/results:
Once get all results in ``` ./output ```
Run all cells in:
```
./src/visualize.ipynb
```
The figures/results should be reproduced. Cheers! 


## 4. Please kindly support us by citing the following papers:
```
@inproceedings{guo2022dynanom,
  title={Subset Node Anomaly Tracking over Large Dynamic Graphs},
  author={Guo, Xingzhi and Zhou, Baojian and Skiena, Steven},
  year={2022}, 
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery & Data Mining}, 
  series = {KDD '22},
  doi = {10.1145/3534678.3539389}, 
  publisher = {Association for Computing Machinery}, 
  url = {https://doi.org/10.1145/3534678.3539389}, 
}

@inproceedings{guo2021subset,
  title={Subset Node Representation Learning over Large Dynamic Graphs},
  author={Guo, Xingzhi and Zhou, Baojian and Skiena, Steven},
  year={2021}, 
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining}, 
  pages = {516–526}, 
  numpages = {11}, 
  series = {KDD '21},
  doi = {10.1145/3447548.3467393}, 
  publisher = {Association for Computing Machinery}, 
  url = {https://doi.org/10.1145/3447548.3467393}, 
}

@inproceedings{zhang2016approximate,
  title={Approximate personalized pagerank on dynamic graphs},
  author={Zhang, Hongyang and Lofgren, Peter and Goel, Ashish},
  booktitle={Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining},
  pages={1315--1324},
  year={2016}
}

@inproceedings{andersen2006local,
  title={Local graph partitioning using pagerank vectors},
  author={Andersen, Reid and Chung, Fan and Lang, Kevin},
  booktitle={2006 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS'06)},
  pages={475--486},
  year={2006},
  organization={IEEE}
}
```
