AirCP
=====
AirCP, standing for *A*uxiliary *I*nformation *R*egularized *CP* tensor completion, is a system for performing CP tensor completion with  regularized trace of the auxiliary information. The Alternating Direction Method of Multipliers (ADMM) is applied to solve the whole optimization problem. 

For details of the algorithm and implementation, see [1], and if you are interested in this direction of research you can check
out our follow up work in [2].

Running
======
The implementation of AirCP has been tested on Matlab 2013. An example of running can be found in `main.m`. [Matlab Tensor Toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html) is necessary and has been included in the package. Synthetic data can be generated with `GenerateSyntheticData.m`.

References
====

[1] Hancheng Ge, James Caverlee, Nan Zhang and Anna Squicciarini. "Uncovering the Spatio-Temporal Dynamics of Memes in the Presence of Incomplete Information". The 25th ACM International Conference on Information and Knowledge Management (CIKM 2016), Indianapolis, USA. [PDF](http://students.cse.tamu.edu/hge/papers/Ge_cikm16.pdf) 

[2] Hancheng Ge, James Caverlee, and Haokai Lu. "TAPER: A Contextual Tensor-Based Approach for Personalized Expert Recommendation". The 10th ACM Conference on Recommender Systems (RecSys 2016), Boston, USA. 
[PDF](http://students.cse.tamu.edu/hge/papers/Ge_recsys16.pdf)
