
# pecok: python3 package for clustering based on PECOK estimator of Bunea, Giraud, Royer, Verzelen ('17)

**Author**: Martin Royer

# Description

pecok is a python package containing a few clustering algorithms (unsupervised learning) for a given number of clusters. It is based on the PECOK paper (https://arxiv.org/abs/1606.05100)

# Installing pecok

The pecok package requires:

* python [>=3.5]
* numpy
* scikit-learn

You can find it on PyPI and install it with:
```shell
pip install pecok
```


### Clustering algorithms

Currently available algorithms are:

  * **pecok_clustering**: the main clustering algorithm described in [Bunea, Giraud, Royer, Verzelen ('17)]

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**obs** | data matrix (n \times p), clustering is applied to the lines of **obs**.|
    |**n_struct** | number of structures to separate the data into.|
    |**int_corr** = 4| correction to be used, between 0 and 4. 0 means no correction, 4 is the correction from [Bunea, Giraud, Royer, Verzelen ('17)]. 1, 2 and 3 are more efficient proxy for the correction, we only recommend 2 and 3. |
    |**rho** = 5| bias-variance tradeoff parameter in ADMM.|
    |**n_iter_max** = -1| if positive, sets the stop condition for maximum number of iteration of the ADMM algorithm used to approximate the solution of the SDP.|
	|**verbose** = False| yields print for time and residuals value at ADMM stop time.|

  * **kmeanz_clustering**: efficient variant for main clustering algorithm, see my PhD thesis

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**obs** | data matrix (n \times p), clustering is applied to the lines of **obs**.|
    |**n_struct** | number of structures to separate the data into.|
    |**int_corr** = 4| correction to be used, between 0 and 4. 0 means no correction, 4 is the correction from [Bunea, Giraud, Royer, Verzelen ('17)]. 1, 2 and 3 are more efficient proxy for the correction, we only recommend 2 and 3. |

