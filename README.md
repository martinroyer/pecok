
# pecok: python3 package for clustering based on PECOK estimator of Bunea, Giraud, Royer, Verzelen ('17)

**Author**: Martin Royer

# Description

pecok is a python package containing a few clustering algorithms (unsupervised learning) for a given number of clusters. It is based on the PECOK paper https://arxiv.org/abs/1606.05100 for variable clustering that later got extended here: https://arxiv.org/abs/1508.01939 (to appear in Annals of Statistics). See also https://papers.nips.cc/paper/6776-adaptive-clustering-through-semidefinite-programming.

# Installing pecok

The pecok package requires:

* python [>=3.5]
* numpy
* scikit-learn

You can find it on PyPI and install it with:
```shell
pip install pecok
```

# Minimal example

```python
    
import numpy as np
np.random.seed(232123123)
n = 10
p = 100
X = np.zeros((n, p))
snr = 0.3
X[n//2:, :] = -np.ones(p) * snr + np.random.normal(scale=.1, size=(n//2,p))
X[:n//2, :] = np.ones(p) * snr + np.random.normal(scale=1, size=(n//2,p))

from pecok import Pecok, KMeanz
Pecok(n_clusters=2, corr=4).fit(X).labels_
KMeanz(n_clusters=2, corr=2).fit(X).labels_
```


## Clustering algorithms

Currently available algorithms (with sklearn object framework implementing `fit` routine) are:

  * **Pecok**: the main clustering algorithm described in [Bunea, Giraud, Royer, Verzelen ('17)]

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**obs** | data matrix (n \times p), clustering is applied to the lines of **obs**.|
    |**n_struct** | number of structures to separate the data into.|
    |**int_corr** = 4| correction to be used, between 0 and 4. 0 means no correction, 4 is the correction from [Bunea, Giraud, Royer, Verzelen ('17)]. 1, 2 and 3 are more efficient proxy for the correction, we only recommend 2 and 3. |
    |**rho** = 5| bias-variance tradeoff parameter in ADMM.|
    |**n_iter_max** = -1| if positive, sets the stop condition for maximum number of iteration of the ADMM algorithm used to approximate the solution of the SDP.|
	|**verbose** = False| yields print for time and residuals value at ADMM stop time.|

  * **KMeanz**: efficient variant for main clustering algorithm, introduced in my PhD thesis: http://www.theses.fr/2018SACLS442

    Parameters:

    | **name** | **description** |
    | --- | --- |
    |**obs** | data matrix (n \times p), clustering is applied to the lines of **obs**.|
    |**n_struct** | number of structures to separate the data into.|
    |**int_corr** = 4| correction to be used, between 0 and 4. 0 means no correction, 4 is the correction from [Bunea, Giraud, Royer, Verzelen ('17)]. 1, 2 and 3 are more efficient proxy for the correction, we only recommend 2 and 3. |

