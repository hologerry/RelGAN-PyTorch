# RelGAN-PyTorch

A **PyTorch** implementation of [**RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes**](https://arxiv.org/abs/1908.07269)

The paper is accepted to ICCV 2019. We also have the Keras version [here](https://github.com/willylulu/RelGAN-Keras).

## Get Started

#### Install

1. Python 3.6 or higher
2. PyTorch 0.4.0 or higher
3. All the dependencies

```bash
pip3 install -r requirements.
```

#### Start TensorBoard server

```bash
tensorboard --logdir runs
```

#### Train your RelGAN!

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --data <PATH_TO_CELEBA-HQ> --gpu [--image_size 256]
```

#### Use multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --data <PATH_TO_CELEBA-HQ> --multi_gpu [--image_size 256]
```

#### Specify your own training settings in `config.yaml`

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --config config.yaml
```



# Experiments log


## Original RelGAN
commit: 1b4b877

### Problems
(iterations: 26000)
0. val_img_xxx: validation results
    revealed problems:
    a. reconstruction is good but not attribute transferring
    b. interpolation is basicly not working

1. test_img: testing the attribute modification
    tested attributes
    ```python
    test_attributes = [
        ('Black_Hair', 1), ('Blond_Hair', 1), ('Brown_Hair', 1),
        ('Male', 1), ('Male', -1), ('Mustache', 1), ('Pale_Skin', 1),
        ('Smiling', 1), ('Bald', 1), ('Eyeglasses', 1), ('Young', 1), ('Young', -1)
    ]
    ```
    problems revealed by results:
    a. artifacts
    b. hair color cannot be changed, or wrong
    c. the last few attributes cannot even be changed at all

2. test_img_iter: testing the interpolation
    tested attributes
    ```python
    ['Smiling', 'Young', 'Mustache']
    ```
    problems revealed by results:
    a. 'Smiling', most images dont work, only a few ones work, some are mutations
    b. 'Young' and 'Mustache', basicly not working



## Cat attribtue z to resblock and upsample layers
commit: 56ba589
