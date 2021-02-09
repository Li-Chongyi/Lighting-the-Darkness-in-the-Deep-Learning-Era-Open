## Lighting the Darkness in the Deep Learning Era: A Survey, An Online Platform, A New Dataset
This repository provides a unified online platform, **DarkPlatform**, that covers many popular deep learning-based LLIE methods, of which the results can be produced through a user-friend web interface, contains a low-light image and video dataset, **LLIVPhone**, in which the images and videos are taken by various phones' cameras under diverse illumination conditions, and collects deep learning-based low-light image and video enhancement **methods, datasets, and evaluation metrics**. More content and details can be found in our Survey Paper: [Lighting the Darkness in the Deep Learning Era]().

✌We will periodically update the content.  It is welcome to let us know if we miss your work that is published in top-tier Journal or conference. We will add it.


## Contents
1. [DarkPlatform](#DarkPlatform)
3. [LLIVPhone Dataset](#LLIVPhone)
4. [Methods](#Methods)
5. [Datasets](#Datasets)
6. [Metrics](#Metrics)
7. [Citation](#Citation)

### DarkPlatform
Currently, the DarkPlatform covers 13 popular deep learning-based LLIE methods including LLNet, LightenNet, Retinex-Net, EnlightenGAN, MBLLEN, KinD, KinD++, TBEFN, DSLR, DRBN, ExCNet, Zero-DCE, and  RRDNet,  where the results of any inputs can be produced through a user-friend web interface. Have fun: [DarkPlatform]().

### LLIVPhone
![Overview](/dataset_samples.png)
LLIVPhone dataset contains 78 videos (23,631 images) taken by 11 different phones' cameras including iPhone 6s, iPhone 7, iPhone7 Plus, iPhone8 Plus, iPhone 11, iPhone 11 Pro, iPhone XS, iPhone XR, iPhone SE, Mi 9, OnePlus 5T under diverse illumination conditions (e.g., weak illumination, underexposure, dark, extremely dark, back-lit, non-uniform light, color light sources, etc.) in the indoor and outdoor scenes. The images and videos in LLIVPhone dataset are saved in PNG and MOV formats, respectively.  Anyone can access the [LLIVPhone dataset](). 

### Methods
![Overview](/chronology.png)
|Date|Publication|Title|Abbreviation|Code|Platform|
|---|---|---|---|---|---|
|2017|PR|LLNet: A deep autoencoder approach to natural low-light image enhancement [paper](https://www.sciencedirect.com/science/article/abs/pii/S003132031630125X)|LLNet|[Code](https://github.com/kglore/llnet_color)|Theano|
|2018|PRL|LightenNet: A convolutional neural network for weakly illuminated image enhancement [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865518300163)|LightenNet|[Code](https://li-chongyi.github.io/proj_lowlight.html)|Caffe & Matlab|
|2018|BMVC|Deep retinex decomposition for low-light enhancement [paper](https://arxiv.org/abs/1808.04560)|Retinex-Net|[Code](https://github.com/weichen582/RetinexNet)|Tensorflow|
|2018|BMVC|MBLLEN: Low-light image/video enhancement using CNNs [paper](http://bmvc2018.org/contents/papers/0700.pdf)|MBLLEN|[Code](https://github.com/Lvfeifan/MBLLEN)|Tensorflow|
|2018|TIP|Learning a deep single image contrast enhancer from multi-exposure images [paper](https://ieeexplore.ieee.org/abstract/document/8259342/)|SCIE|[Code](https://github.com/csjcai/SICE)|Caffe & Matlab|
|2018|CVPR|Learning to see in the dark [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Learning_to_See_CVPR_2018_paper.html)|Chen et al.|[Code](https://github.com/cchen156/Learning-to-See-in-the-Dark)|Tensorflow|
|2018|NeurIPS|DeepExposure: Learning to expose photos with asynchronously reinforced adversarial learning [paper](https://dl.acm.org/doi/abs/10.5555/3326943.3327142)|DeepExposure| |Tensorflow|
|2019|ICCV|Seeing motion in the dark [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Seeing_Motion_in_the_Dark_ICCV_2019_paper.html)|Chen et al.|[Code](https://github.com/cchen156/Seeing-Motion-in-the-Dark)|Tensorflow|
|2019|ICCV|Learning to see moving object in the dark [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Jiang_Learning_to_See_Moving_Objects_in_the_Dark_ICCV_2019_paper.html)|Jiang and Zheng|[Code](https://github.com/MichaelHYJiang)|Tensorflow|
|2019|CVPR|Underexposed photo enhancement using deep illumination estimation [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Underexposed_Photo_Enhancement_Using_Deep_Illumination_Estimation_CVPR_2019_paper.html)|DeepUPE|[Code](https://github.com/Jia-Research-Lab/DeepUPE)|Tensorflow|
|2019|ACMMM|Kindling the darkness: A practical low-light image enhancer [paper](https://dl.acm.org/doi/abs/10.1145/3343031.3350926)|KinD|[Code](https://github.com/zhangyhuaee/KinD)|Tensorflow|
|2019|ACMMM|Kindling the darkness: A practical low-light image enhancer [paper](https://dl.acm.org/doi/abs/10.1145/3343031.3350926)|KinD // KinD++|[Code](https://github.com/zhangyhuaee/KinD)|Tensorflow|
