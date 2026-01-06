# CLIP [JAX+FLAX]

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach

![CLIP](CLIP.png)


## Details
The ViT model and checkpoints have been ported to Haiku, while preserving the same output. See tests/test_consistency.py for details.

No JIT/pmap is performed, but pure inference functions for both the text and image encoders are provided from the `clip_jax.load()` function, which should be easy to run/parallelize how you wish.

## Usage

First, [install jax 0.8.2](https://docs.jax.dev/en/latest/installation.html), [flax 0.7.5](https://flax.readthedocs.io/en/v0.8.3/quick_start.html#install-flax) (or later), [PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ pip install pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install jax[cuda12]>=0.8.0 flax>=0.8.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/willz-blankOS/CLIP_flax.git
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

```python
from PIL import Image
import jax
import torch
import clip_flax

clip, preprocess = clip.load("ViT-B/32", device="gpu")

image = preprocess(Image.open("CLIP.png"))
image = image.transpose((0, 2, 3, 1))

text = clip.tokenize(["A dog pissing behind the tree"])

image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits_per_image, logits_per_text = model(image, text)
probs = jax.nn.softmax(logits_per_image, axis=-1).numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```


## API

The CLIP module `clip` provides the following methods:

#### `clip.available_models()`

Returns the names of the available CLIP models.

#### `clip.load(name, device=...)`

Returns the model and the TorchVision transform needed by the model, specified by the model name returned by `clip.available_models()`. It will download the model as necessary. The `name` argument can also be a path to a local checkpoint.

The device to run the model can be optionally specified, and the default is to use the first CUDA device if there is any, otherwise the CPU.

#### `clip.tokenize(text: Union[str, List[str]], context_length=77)`

Returns a Array containing tokenized sequences of given text input(s). This can be used as the input to the model

---

The model returned by `clip.load()` supports the following methods:

#### `model.encode_image(image: Tensor)`

Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.

#### `model.encode_text(text: Array)`

Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.

#### `model(image: Array, text: Array)`

Given a batch of images and a batch of text tokens, returns two Arrays, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.
