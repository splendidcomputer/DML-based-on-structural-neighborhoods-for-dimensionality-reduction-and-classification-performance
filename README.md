# Distance metric learning based on structural neighborhoods for dimensionality reduction and classification performance improvement

This implementation first tries to extract a low dimensional manifold from the input data. Then, it learns the local neighborhood structures and the relationship of the data points in the ambient space based on the adjacencies of the same data points on the embedded low dimensional manifold. Using the local neighborhood relationships extracted from the manifold space, the proposed method learns the distance metric in a way which minimizes the distance between similar data and maximizes their distance from the dissimilar data points.

## Paper:

This source code forms the basis for our [Distance metric learning based on structural neighborhoods for dimensionality reduction and classification performance improvement](https://arxiv.org/abs/1902.03453). When using the code, please make sure to refer to it as follows:

```
@misc{https://doi.org/10.48550/arxiv.1902.03453,
  doi = {10.48550/ARXIV.1902.03453},
  
  url = {https://arxiv.org/abs/1902.03453},
  
  author = {Ghods, Mostafa Razavi and Moattar, Mohammad Hossein and Forghani, Yahya},
  
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Distance metric learning based on structural neighborhoods for dimensionality reduction and classification performance improvement},
  
  publisher = {arXiv},
  
  year = {2019},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Requirements
The code was tested with Matlab R2017a, 64 Bit under Windows 10 using Intel® Core™ i7-2670QM 2.20 GHz with 8 GB RAM.

## Execution

In order to run this code you have to add the [DR Toolbox](https://lvdmaaten.github.io/drtoolbox/) to the Matlab path. Then, you can execut the program by running the  **_main.m_** file.
