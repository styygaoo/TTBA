# Introduction

This is a project that has provided an example to implement a test-time batch adaptation (TTBA) Algorithm [1] on a depth estimation model [2].

The code of TENT [3] helped a lot as well. 


## **Environment setup**

Install environment using `environment.yml` : 

Using conda : 

```bash
conda env create -n TTBA --file environment.yml
conda activate TTBA
```

Suppose you can not install the environment TTBA via the above command. In that case, you can install the required libraries one by one manually since there are only few additional libraries needed to run this program.

The project depends on :
- [pytorch](https://pytorch.org/) (Main framework)
- PIL


## **Usage**

the main.py is the file to run the TTBA on the model Guided decoding model(GDM) from the source domain KITTI to the target domain VKITTI.

To check the performance of the original model, you can run test_original_model.py

For Windows users, this line 

testset_loader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)       # drop_last=True

must be modified to

testset_loader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)       # drop_last=True


## Reference

[1] S. Zhang, L. Yang, M. Bi Mi, X. Zheng, A. Yao, "Improving Deep Regression with Ordinal Entropy," in ICLR, 2023.

[2] Rudolph, M., Dawoud, Y., Güldenring, R., Nalpantidis, L., & Belagiannis, V. (2022, May). Lightweight monocular depth estimation through guided decoding. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 2344-2350). IEEE.

[3] Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2020). Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726.

