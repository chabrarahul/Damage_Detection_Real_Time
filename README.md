# Damage_Detection_Real_Time


Unet model was implemented on real world damage detection dataset(Semantic Segmentation). 

Mobile architecture such as MobileNet_V2 as encoder and DeepLab_V3 as decoder was also implemented to detect damages in real time. 


## Pruning, quantization and Sparsify of the above were done to increase the inferece time and reduce the computational cost. So that it can be run on embedded devise such as Jetson Nano and Jetson TX2


Refernces 

Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.


Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).


