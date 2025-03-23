# NTIRE2025
## Pureformer: Transformer-Based Image Denoising
#### Proposed Pureformer encoder-decoder architecture for image denoising. The input noisy image is processed through a multi-level encoder, a feature enhancer block, and a multi-level decoder. Each encoder and decoder level employs xN transformer blocks, consisting of Multi-Dconv Head Transposed Attention (MDTA) and Gated-Dconv Feed-Forward Network (GDFN) blocks. The feature enhancer block, placed in the latent space, expands the receptive field using a spatial filter bank. The multi-scale features are then concatenated and refined through xN transformer blocks to enhance feature correlation and merge multi-scale information effectively.
![image](https://github.com/user-attachments/assets/b5d55bcb-aadd-41a5-8f6e-1ddd49f16853)

go to the dir NTIRE2025_cipher_vision

``` cd NTIRE2025_cipher_vision/```

install the env the conda env:

```conda env create -f nr_pf.yml```

activate it using:

```conda activate nr_pf.env```

run this command first:

```pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118```

run requiements:
 
``` pip install -r requirements.txt --no_deps ``` 

run for hetting image results in "results/"

```python test_demo.py```

## installation process:
1. add images in data/input/noisy/ (if not create one)
2. pretained model = `model_zoo`
3. model file = `36_Pureformer.py`
4. download "36_Pureformer.ckpt" from releases and put it dir `model_zoo`
5. options for managing pretained_model path, input image path, output_path are given from line no. 160 in test_demo.py
