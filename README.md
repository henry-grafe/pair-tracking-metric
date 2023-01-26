# pair-tracking-metric

This is the code for generating the plots for my master thesis, "Metric definition and optimization for appearance information in Multi-Object Tracking".

The extracted features used for some of the plot can be downloaded [here](https://drive.google.com/drive/folders/1sD1d9A6Tc2GTHLta6DBBSS4_doATUsCl?usp=sharing) and where extracted from the test set of [Market-1501](https://paperswithcode.com/paper/scalable-person-re-identification-a-benchmark) using [this model](https://arxiv.org/abs/2211.03679) using the [torchreid](https://github.com/KaiyangZhou/deep-person-reid) framework. You need to put the two files into the same folder, and pass the name of the folder as an argument in the scripts.

Requirements :
- Pickle
- Numpy
- Scipy
- Matplotlib
- Pytorch 1.10.1
