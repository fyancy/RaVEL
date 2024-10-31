# RaVEL
![](https://img.shields.io/badge/language-python-orange.svg)
[![](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/fyancy/MetaFD/blob/main/LICENSE)
[![](https://img.shields.io/badge/CSDN-燕策西-blue.svg)](https://blog.csdn.net/weixin_43543177?spm=1001.2101.3001.5343)
[![](https://img.shields.io/badge/Homepage-YongFeng-purple.svg)](https://fyancy.github.io/)


Fast Random Wavelet Convolution for Weak-Fault Diagnosis

**The paper entitled [_Beyond deep features: Fast random wavelet kernel convolution for weak-fault feature extraction of rotating machinery_](https://www.sciencedirect.com/science/article/pii/S0888327024009555?dgcid=coauthor) has been published on MSSP. Authors are sorting out the code and and will publish a complete version soon. Please wait for a moment.** 🙃 🗓 By the end of this week.

- [x] Quick Start
- [ ] Proposed method and its components
- [ ] CPU-based comparison methods
- [ ] GPU-based comparison methods

[2024-10-30] We have provided a [`quick start`](https://github.com/fyancy/RaVEL/blob/main/quick_start.ipynb) file to fast validate RaVEL with open SEU data. Plz download the code file, [data file](https://drive.google.com/drive/folders/1GbioYlKtaTG1KRg9b_Krq3p7z00L7pz7?usp=sharing), and try it.


## Performance 🚀
<div align=center>
<img src="/figs/rl_acc.png" width="500">
</div>
<p align="center">
Fig. 1. Comparison of the proposed method with SOTA methods for EFD. 
</p>

## Structure 🔑
<div align=center>
<img src="/figs/structure_comparison_v2.jpg" width="900">
</div>
<p align="center">
Fig. 2. Illustration of mechanical fault feature extraction methods. (a) Time-frequency features, (b) deep or adaptive
features, (c) hybrid features, and (d) the proposed method.
</p>

## Interpretability
As done in work [Time-Frequency Network](https://github.com/ChenQian0618/TFN) (TFN), the amplitude-frequency response (AFR) can be used to analyze the filter property of kernels and further compare it with the FFT of signals. 
<div align=center>
<img src="/figs/TFN_sq_snr_10.png" width="400">
</div>
<p align="center">
Fig. AFR of time-freq kernels. This is the reproduced TFN, which concentrates on the global frequencies.
</p>

The proposed method could generate a lot of good **low-pass filters**, which is just in line with the fact that faults exist in the low frequency region.
|Fig. 1. Results on normal sample   | Fig. 2. Results on inner fault sample  | Fig. 3. Results on outer fault sample  |
|:----:|:----:|:----:|
|<img src="/figs/sample2.jpg" width="300" /><br/> | <img src="/figs/sample68.jpg" width="300" /><br/>| <img src="/figs/sample250.jpg" width="300" /><br/>|

## Acknowledgements
Thanks to the following open source codes and the researchers:
- [Time-Frequency Network](https://github.com/ChenQian0618/TFN)
- [WaveletKernelNet](https://github.com/HazeDT/WaveletKernelNet)
- [ROCKET](https://github.com/angus924/rocket) and [MultiROCKET](https://github.com/ChangWeiTan/MultiRocket)



[![Yong Feng's GitHub stats](https://github-readme-stats.vercel.app/api?username=fyancy&show_icons=true&theme=gruvbox)](https://github.com/fyancy)
