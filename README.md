# How to Contribute to the ImageNet Benchmark

## PyTorch

1. Install the torchbench library:

```bash
pip install torchbench
```

2. In the root of your model repository, make sure you have a `requirements.txt` file specifying the dependencies for your project, and including:

```
Pillow
scipy
torch
torchbench
torchvision
```

3. Write a `benchmark.py` file in the root of your project. You can use the following as a template:

```python
from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import PIL

# Define Transforms    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# My Model (replace this with your model - import from your project to get the model object)
my_model = ...

# Run Evaluation (Example metadata for EfficientNet)
ImageNet.benchmark(
    model=my_model,
    paper_model_name='EfficientNet-B0',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)
```

If you need complete examples, checkout these benchmark files:

- [EfficientNet benchmark.py]()
- [ResNet benchmark.py]()
- [DPN benchmark.py]()
