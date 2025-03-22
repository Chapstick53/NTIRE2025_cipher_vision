# NTIRE2025
## Pureformer: Transformer-Based Image Denoising
![image](https://github.com/user-attachments/assets/b5d55bcb-aadd-41a5-8f6e-1ddd49f16853)

install the env the conda env:

```conda env create -f nr_pf.yml```

activate it using:

```conda activate nr_pf.env```

run requiements:
 
``` pip install -r requirements.txt ``` 

run for hetting image results in "results/"

```python test_demo.py```

installation process:
1. add images in data/input/noisy/ (if not create one)
2. pretained model = model_zoo
3. model file = model.py
4. download "36_Pureformer.ckpt" from releases and put it dir `model_zoo`
5. options for managing pretained_model path, input image path, output_path are given from line no. 160 in test_demo.py
