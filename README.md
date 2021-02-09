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
|2019|ACMMM (IJCV)|Kindling the darkness: A practical low-light image enhancer [paper](https://dl.acm.org/doi/abs/10.1145/3343031.3350926) (Beyond brightening low-light images [paper](https://link.springer.com/article/10.1007/s11263-020-01407-x))|KinD (KinD++)|[Code](https://github.com/zhangyhuaee/KinD)|Tensorflow|
|2019|ACMMM|Progressive retinex: Mutually reinforced illumination-noise perception network for low-light image enhancement [paper](https://dl.acm.org/doi/abs/10.1145/3343031.3350983)|Wang et al.| |Caffe|
|2019|TIP|Low-light image enhancement via a deep hybrid network [paper](https://ieeexplore.ieee.org/abstract/document/8692732/)|Ren et al.| |Caffe|
|2019(2021)|arXiv(TIP)|EnlightenGAN: Deep light enhancement without paired supervision [paper](https://ieeexplore.ieee.org/abstract/document/9334429/)  [arxiv](https://arxiv.org/pdf/1906.06972.pdf)|EnlightenGAN|[Code](https://github.com/VITA-Group/EnlightenGAN) |Pytorch|
|2019|ACMMM|Zero-shot restoration of back-lit images using deep internal learning [paper](https://dl.acm.org/doi/abs/10.1145/3343031.3351069)|ExCNet|[Code](https://cslinzhang.github.io/ExCNet/)|Pytorch|
|2020|CVPR|Zero-reference deep curve estimation for low-light image enhancement [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html)|Zero-DCE|[Code](https://github.com/Li-Chongyi/Zero-DCE)|Pytorch|
|2020|CVPR|From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.html)|DRBN|[Code](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)|Pytorch|
|2020|ACMMM|Fast enhancement for non-uniform illumination images using light-weight CNNs [paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413925)|Lv et al.| |Tensorflow|
|2020|ACMMM|Integrating semantic segmentation and retinex model for low light image enhancement [paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413757)|Fan et al.| | |
|2020|CVPR|Learning to restore low-light images via decomposition-and-enhancement [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Learning_to_Restore_Low-Light_Images_via_Decomposition-and-Enhancement_CVPR_2020_paper.html)|Xu et al.| |Pytorch|
|2020|AAAI|EEMEFN: Low-light image enhancement via edge-enhanced multi-exposure fusion network [paper](https://ojs.aaai.org/index.php/AAAI/article/view/7013)|EEMEFN| |Pytorch|
|2020|TIP|Lightening network for low-light image enhancement [paper](https://ieeexplore.ieee.org/abstract/document/9141197)|DLN| |Pytorch|
|2020|TMM|Luminance-aware pyramid network for low-light image enhancement [paper](https://ieeexplore.ieee.org/abstract/document/9186194)|LPNet| |Pytorch|
|2020|ECCV|Low light video enhancement using synthetic data produced with an intermediate domain mapping [paper](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_7)|SIDGAN| |Tensorflow|
|2020|TMM|TBEFN: A two-branch exposure-fusion network for low-light image enhancement [paper](https://ieeexplore.ieee.org/abstract/document/9261119/)|TBEFN|[Code](https://github.com/lukun199/TBEFN) |Tensorflow|
|2020|ICME|Zero-shot restoration of underexposed images via robust retinex decomposition [paper](https://ieeexplore.ieee.org/abstract/document/9102962/)|RRDNet|[Code](https://aaaaangel.github.io/RRDNet-Homepage) |Pytorch|
|2020|TMM|DSLR: Deep stacked laplacian restorer for low-light image enhancement [paper](https://ieeexplore.ieee.org/abstract/document/9264763/)|DSLR|[Code](https://github.com/SeokjaeLIM/DSLR-release) |Pytorch|


### Datasets
|Abbreviation|Number|Format|Real/Synetic|Video|Paired/Unpaired/Application|Dataset|
|---|---|---|---|---|---|---|
|LOL [paper](https://arxiv.org/abs/1808.04560)|500|RGB|Real|No|Paired|[Dataset](https://daooshee.github.io/BMVC2018website/)|
|SCIE [paper](https://ieeexplore.ieee.org/abstract/document/8259342/)|4413|RGB|Real|No|Paired|[Dataset](https://github.com/csjcai/SICE)|
|MIT-Adobe FiveK [paper](http://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf)|5000|Raw|Real|No|Paired|[Dataset](https://data.csail.mit.edu/graphics/fivek/)|
|SID [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Learning_to_See_CVPR_2018_paper.html)|5094|Raw|Real|No|Paired|[Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)|
|DRV [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Seeing_Motion_in_the_Dark_ICCV_2019_paper.html)|202|Raw|Real|Yes|Paired|[Dataset](https://github.com/cchen156/Seeing-Motion-in-the-Dark) |
|SMOID [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Jiang_Learning_to_See_Moving_Objects_in_the_Dark_ICCV_2019_paper.html)|179|Raw|Real|Yes|Paired|[Dataset](https://github.com/MichaelHYJiang) |
|LIME [paper](https://ieeexplore.ieee.org/abstract/document/7782813)|10|RGB|Real|No|Unpaired|[Dataset](https://drive.google.com/file/d/0BwVzAzXoqrSXb3prWUV1YzBjZzg/view)|
|NPE [paper](https://ieeexplore.ieee.org/abstract/document/6512558)|84|RGB|Real|No|Unpaired|[Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|MEF [paper](https://ieeexplore.ieee.org/abstract/document/7120119)|17|RGB|Real|No|Unpaired|[Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|DICM [paper](https://ieeexplore.ieee.org/abstract/document/6467022)|64|RGB|Real|No|Unpaired|[Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|VV|24|RGB|Real|No|Unpaired|[Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|ExDARK [paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314218304296)|7363|RGB|Real|No|Application|[Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)|
|BBD-100K [paper](https://udparty.com/php/upload/20180627/153006477668ddb6563df254a8.pdf)|10,000|RGB|Real|Yes|Application|[Dataset](https://bdd-data.berkeley.edu/)|
|DARK FACE [paper](https://arxiv.org/abs/1904.04474)|6000|RGB|Real|No|Application|[Dataset](https://flyywh.github.io/CVPRW2019LowLight/)|
