Table of Contents
- [Finding Natural Patterns in Data](#finding-natural-patterns-in-data)
  - [1. Low Dimensional Visualisation](#1-low-dimensional-visualisation)
    - [a. Dimensionality Reduction Techniques](#a-dimensionality-reduction-techniques)
    - [b. Multidimensional Scaling (MDS)](#b-multidimensional-scaling-mds)
      - [i. Calculating pairwise distances](#i-calculating-pairwise-distances)
      - [ii. Performing MDS](#ii-performing-mds)
- [Classification Methods](#classification-methods)
- [Improving Predictive Models](#improving-predictive-models)
- [Regression Methods](#regression-methods)
- [Neural Networks](#neural-networks)

# Finding Natural Patterns in Data

The goal of unsupervised learning problems is to identify the natural patterns or groupings in a dataset. As an example, let's examine a table containing information about basketball players:

| *Player* | Height | Weight | Points | Rebounds | Blocks | Assists |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| A | | | | | | |
| B | | | | | | |
| C | | | | | | |
| D | | | | | | |
| E | | | | | | |
| ... | | | | | | |
| Z | | | | | | |
|  | | | | | | |

On what basis can we divide the players? In basketball, players are divided into three groups of positions - Guards; Forwards and Centres. Teams generally rely on these three positions when choosing their players with the assumption that players who play the same position will have similar sets of skills and statistics. *But we don't know that ... yet.*

Back to our ML. With enforcing any prior knowledge about basketball, unsupervised learning can be used to divide the players' statistics into groups. Once we divide the players into groups, we can then examine whether these clusters correspond to the conventional positions in basketball.



## 1. Low Dimensional Visualisation
A quick and easy way tosee if the players can be grouped (based on their stats) is to visualise the data and see if there are any obvious trends or patterns. In order to effectively visualise data that contains more than 3 variables, we can use <b>dimensionality reduction techniques</b> such as multidimensional scaling and principal component analysis.

### a. Dimensionality Reduction Techniques
Visualising data in 2D or 3D is simple. However, ML problems involve myriads of dimensions which make visualisation a tricky task. Typically there is an internal structure to that data i.e. most of the information is in fact contained in fewer dimensions than the entirety of the data set. That is, it is possible to transform variables into a lower dimensional space without losing the integrity of information. For relatively small problems, it may be possible to visualise the transformed data in 2D or 3D to get an approximate representation of the data. 

<b>Principal Component Analysis (PCA)</b> and <b>Classical Multidimensional Scaling (cMDS)</b> are the two common ways in which we can potentially reduce the number of dimensions. Both methods are about building a new orthogonal coordinate system where the coordinates are ordered by importance. In PCA, the coordinates are in order of how much variance in the data they explain. It always transforms an <i>n</i>-dimensional space into another <i>n</i>-dimensional space. In MDS, they are ordered by how closely they preserve the pair-wise distances between observations. <i>n</i>-dimensional spaces are always transformed into the smallest space to preserve the pair-wise distances. Because of this, MDS can be used with any distance metric. For example, applying Euclidean space to MDS gives the same result as PCA. 

With both methods, it is possible to get a measure of the importance of each dimension in the new orthogonal coordinate system. This is typically visualised <i>in tandem</i> with a [Pareto chart](https://en.wikipedia.org/wiki/Pareto_chart) which shows each individual value as a bar + the running total as a line. We can use Pareto charts to determine how many dimensions we consider sufficient to obtain a reasonable approximation to the full data. If two or three dimensions are deemed sufficient, then just the first two or three coordinates of the transformed data can be plotted.

### b. Multidimensional Scaling (MDS)

#### i. Calculating pairwise distances

In MATLAB, [`pdist`](https://au.mathworks.com/help/stats/pdist.html) function can be used to calculate the pairwise distance between the observations.

```matlab
% D: distance/dissimilarty vector containing the distance between each pair of
% observations. D is of length m(m-1)/2
%
% data: m x n numeric matrix containing the data. each of the m rows is
% considered an observation
%
% "distance": option input, indicates the method of calculating the distance or
% dissimilarity. options: "euclidean" (default), "cityblock" and "correlation"  
>> D = pdist(data, "distance")
```

#### ii. Performing MDS

Dissimilairity vectors can be used an input to the MATLAB function [`cmdscale`](https://au.mathworks.com/help/stats/cmdscale.html).

```matlab
% x: m x q matrix of the reconstructed coordinates in q-dimensional space
% q is the minimum number of dimensions needed to achieve the given pdist.
%
% e: eigenvalues of the matrix x * x'
% 
% D: see above
>> [x, e] = cmdscale(D)
```

eigenvalues `e` can be used to determine if a low-dimensional approximation to the points in `x` provides a reasonable representation of the dat. IF the first `p` eigenvalues are significantly larger than the rest, the points are well approximated by the first `p` dimensions i.e. the first `p` columns of `x`.

# Classification Methods

# Improving Predictive Models

# Regression Methods

# Neural Networks
