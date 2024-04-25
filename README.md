# RaVEL
![](https://img.shields.io/badge/language-python-orange.svg)
[![](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/fyancy/MetaFD/blob/main/LICENSE)
[![](https://img.shields.io/badge/CSDN-燕策西-blue.svg)](https://blog.csdn.net/weixin_43543177?spm=1001.2101.3001.5343)

Fast Random Wavelet Convolution for Early Fault Diagnosis

**Codes will be FULLY publicly available once the peer review is finished.** 🙃 🗓

## Performance 🚀
<div align=center>
<img src="/figs/rl_acc.png" width="400">
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
<div align=center>
<img src="/figs/TFN_sq_snr_10.png" width="600">
</div>
<p align="center">
Fig. Amplitude-frequency response of Kernels. This is reproduced from https://github.com/ChenQian0618/TFN, which concentrate on the global frequencies.
</p>

Good low-pass filter, because the fault exists in the low frequency region.
|Fig. 1. Results on normal sample   | Fig. 2. Results on inner fault sample  | Fig. 3. Results on outer fault sample  |
|:----:|:----:|:----:|
|<img src="/figs/sample2.jpg" width="300" /><br/> | <img src="/figs/sample68.jpg" width="300" /><br/>| <img src="/figs/sample250.jpg" width="300" /><br/>|

## Acknowledgement
Thanks to the following open source code and its researchers
- https://github.com/ChenQian0618/TFN.
- WKN
