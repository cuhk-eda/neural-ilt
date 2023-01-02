# Neural-ILT
Neural-ILT is an end-to-end learning-based mask optimization tool developed by the research team supervised by Prof. Evangeline F.Y. Young in The Chinese University of Hong Kong (CUHK). Neural-ILT attempts to replace the conventional end-to-end ILT (inverse lithography technology) correction process under a holistic learning-based framework. It conducts on-neural-network ILT correction for the given layout under the guidiance of a partial coherent imaging model and directly outputs the optimized mask at the convergence.

Compared to the conventional academia ILT solutions, e.g., [MOSAIC](https://ieeexplore.ieee.org/document/6881379) (Gao *et al.*, DAC'14) and [GAN-OPC](https://ieeexplore.ieee.org/document/8465816) (Yang *et al.*, TCAD'20), Neural-ILT enjoys:
- much faster ILT correction process (20x ~ 70x runtime speedup)
- better mask printability at convergence
- modular design for easy customization and upgradation
- ...

More details are in the following papers:
* Jiang, Bentian, Lixin Liu, Yuzhe Ma, Hang Zhang, Bei Yu, and Evangeline FY Young. "[Neural-ILT: migrating ILT to neural networks for mask printability and complexity co-optimization](https://ieeexplore.ieee.org/abstract/document/9256592)", in 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pp. 1-9. IEEE, 2020.
* Jiang, Bentian, Xiaopeng Zhang, Lixin Liu, and Evangeline FY Young. "[Building up End-to-end Mask Optimization Framework with Self-training](https://dl.acm.org/doi/abs/10.1145/3439706.3447050)", in Proceedings of the 2021 International Symposium on Physical Design (ISPD), pp. 63-70. 2021.
* Jiang, Bentian, Lixin Liu, Yuzhe Ma, Bei Yu, and Evangeline FY Young. "[Neural-ILT 2.0: Migrating ILT to Domain-specific and Multi-task-enabled Neural Network.](https://ieeexplore.ieee.org/abstract/document/9526856)" IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (2021).

## Requirements
-   python: [3.7.3](https://www.python.org/downloads/)
-   pytorch: [1.8.0](https://pytorch.org/)
-   torchvision: [0.2.2](https://pytorch.org/)
-   cudatoolkit: [11.1.1](https://developer.nvidia.com/cuda-toolkit)
-   pillow: [6.1.0](https://pypi.org/project/Pillow/)
-   GPU: >= 10GB GPU memory for pretrain, >= 7GB for Neural-ILT
-   [This repo passes the test on a linux machine of Ubuntu 18.04.6 LTS (GNU/Linux 4.15.0-158-generic x86_64) & CUDA Version: 11.4]

## Usage
**Step 1:** Download the source codes. For example,
~~~bash
$ git clone https://github.com/cuhk-eda/neural-ilt.git
~~~

**Step 2:** Go to the project root and unzip the environment
~~~bash
$ cd Neural-ILT/
$ unzip env.zip
~~~
(Optional) To replace the ICCAD'20 training dataset with the ISPD'21 training dataset (last batch)
~~~bash
$ cd Neural-ILT/dataset/
$ unzip ispd21_train_dataset.zip
~~~

**Step 3:** Conduct Neural-ILT on [ICCAD 2013 mask optimization contest benchmarks](https://ieeexplore.ieee.org/document/6691131)
~~~bash
$ cd Neural-ILT/
$ python neural_ilt.py
~~~
Note that we observed minor variation (±0.5%) on mask printability score (L2+PVB, [statistics of 50 rounds](./stat/variation_test_neural_ilt.xlsx)). We haven't yet located the source of non-determinism. We would appreciate any insight from the community for resovling this non-determinism :sparkles:.

**Step 4 (optional):** Backbone model pre-training
~~~bash
$ cd Neural-ILT/
$ python pretrain_model.py
~~~

**Evaluation:** Evaluate the mask printability
~~~bash
$ cd Neural-ILT/
$ python eval.py --layout_root [root_to_layout_file] --layout_file_name [your_layout_file_name] --mask_root [root_to_mask_file] --mask_file_name [your_mask_file_name]
~~~

### Parameters
```angular2html
 |── neural_ilt.py
 |   ├── device/gpu_no: the device id
 |   ├── load_model_name/ilt_model_path: the pre-trained model of Neural-ILT
 |   ├── lr: initial learning rate
 |   ├── refine_iter_num: maximum on-neural-network ILT correction iterations
 |   ├── beta: hyper-parameter for cplx_loss in the Neural-ILT objective
 |   ├── gamma: lr decay rate
 |   ├── step_size: lr decay step size
 |   ├── bbox_margin: the margin of the crop bbox
 |
 |── pretrain_model.py
 |   ├── gpu_no: the device id
 |   ├── num_epoch: number of training epochs
 |   ├── alpha: cycle loss weight for l2
 |   ├── beta: cycle loss weight for cplx
 |   ├── lr: initial learning rate
 |   ├── gamma: lr decay rate
 |   ├── step_size: lr decay step size
 |   ├── margin: the margin of the crop bbox
 |   ├── read_ref: read the pre-computed crop bbox for each layout
 |
 |── End
```

Expolre your own recipe for model pretraining and Neural-ILT. Have fun! :smile:


## Acknowledgement
We wolud like to thank the authors of [GAN-OPC](https://ieeexplore.ieee.org/document/8465816) (Yang *et al.*, TCAD'20) for providing the [training layouts](https://github.com/phdyang007/GAN-OPC) used in our ICCAD'20 paper. Based on which, we further generated the ISPD'21 training layouts following the procedure described in [Jiang *et al.*, ISPD'21](https://dl.acm.org/doi/abs/10.1145/3439706.3447050).


## Contact
[Bentian Jiang](https://infamousmega.github.io/) (btjiang@cse.cuhk.edu.hk) and [Lixin Liu](https://liulixinkerry.github.io) (lxliu@cse.cuhk.edu.hk)



## Citation
If Neural-ILT is useful for your research, please consider citing the following papers:
```angular2html
@inproceedings{jiang2020neural,
  title={Neural-ILT: migrating ILT to neural networks for mask printability and complexity co-optimization},
  author={Jiang, Bentian and Liu, Lixin and Ma, Yuzhe and Zhang, Hang and Yu, Bei and Young, Evangeline FY},
  booktitle={2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD)},
  pages={1--9},
  year={2020},
  organization={IEEE}
}
@inproceedings{jiang2021building,
  title={Building up End-to-end Mask Optimization Framework with Self-training},
  author={Jiang, Bentian and Zhang, Xiaopeng and Liu, Lixin and Young, Evangeline FY},
  booktitle={Proceedings of the 2021 International Symposium on Physical Design},
  pages={63--70},
  year={2021}
}
@article{jiang2021neural,
  title={Neural-ILT 2.0: Migrating ILT to Domain-specific and Multi-task-enabled Neural Network},
  author={Jiang, Bentian and Liu, Lixin and Ma, Yuzhe and Yu, Bei and Young, Evangeline FY},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2021},
  publisher={IEEE}
}
```



## License
READ THIS LICENSE AGREEMENT CAREFULLY BEFORE USING THIS PRODUCT. BY USING THIS PRODUCT YOU INDICATE YOUR ACCEPTANCE OF THE TERMS OF THE FOLLOWING AGREEMENT. THESE TERMS APPLY TO YOU AND ANY SUBSEQUENT LICENSEE OF THIS PRODUCT.



License Agreement for Neural-ILT



Copyright (c) 2021, The Chinese University of Hong Kong
All rights reserved.



CU-SD LICENSE (adapted from the original BSD license) Redistribution of the any code, with or without modification, are permitted provided that the conditions below are met. 



1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.



2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.



3. Neither the name nor trademark of the copyright holder or the author may be used to endorse or promote products derived from this software without specific prior written permission.



4. Users are entirely responsible, to the exclusion of the author, for compliance with (a) regulations set by owners or administrators of employed equipment, (b) licensing terms of any other software, and (c) local, national, and international regulations regarding use, including those regarding import, export, and use of encryption software.



THIS FREE SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR ANY CONTRIBUTOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, EFFECTS OF UNAUTHORIZED OR MALICIOUS NETWORK ACCESS; PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

