# Understanding the Representation Power of Graph Neural Networks in Learning Graph Topology

Code for NeurIPS 2019 paper titled [Understanding the Power of Graph Neural Networks in Learning Graph Topology](https://arxiv.org/abs/1907.05008)

Code is written with Python 3.6.5.  

### Poster, Slides and video

The poster and slides can be found in ``doc/``.  

The video can be found [here](https://www.youtube.com/watch?v=kk_x0wOvZYQ).  
 

### Set up 

1. `git clone https://github.com/nimadehmamy/Understanding-GCN.git`
2. `pip install -r requirements.txt`
 

### Source Files

-  ``GraphConvNet.py``

Code for modular design of the graph convolutional networks 



### Notebooks

-  ``gcn-classification``

Notebook for Graph Stethoscope experiments

- ``gcn-moments-experiments``

Notebook for validating graph moment learning theory

- ``GCN-vs-FC-graph-moments`` 

Notebook for tests comparing a fully-connected layer with GCN for learning graph moments

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@article{dehmamy2019understanding,
  title={Understanding the Representation Power of Graph Neural Networks in Learning Graph Topology},
  author={Dehmamy, Nima and Barab{\'a}si, Albert-L{\'a}szl{\'o} and Yu, Rose},
  journal={Advances in neural information processing systems},
  year={2019}
}
```
