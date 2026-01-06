import hashlib
import os
import urllib
import warnings
import numpy as np
from packaging import version
from typing import Callable, Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

import jax
import jax.numpy as jnp

import flax.nnx as nnx

from model import CLIP, get_params
from simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def convert_params(torch_state, jax_params, rn=False):
    def name_iter(pytree, root, f):
        new_out = {}
        for k, v in pytree.items():
            if rn:
                if 'visual' not in k:
                    if isinstance(v, dict):
                        new_out[k] = name_iter(v, root + "." + str(k), f)
                    else:
                        new_out[k] = f(v, root + "." + str(k))
            else:
                if isinstance(v, dict):
                    new_out[k] = name_iter(v, root + "." + str(k), f)
                else:
                    new_out[k] = f(v, root + "." + str(k))
        return new_out

    def process_node(value, name):
        name = name.lstrip(".")
        tensor_name = name.split(".")[-1]
        tensor_name = {
            "kernel": "weight",
            "scale": "weight",
            "embedding": "weight",
        }.get(tensor_name, tensor_name)

        tensor_path = ".".join(name.split(".")[:-1])
        new_tensor = value

        pytorch_name = f"{tensor_path}.{tensor_name}" if tensor_path != "" else tensor_name
        #print(tensor_path + "." + tensor_name)
        if "conv" in name:
            pytorch_tensor = torch_state[pytorch_name].permute([2, 3, 1, 0])
            new_tensor = jnp.array(pytorch_tensor)
        elif pytorch_name in torch_state:
            pytorch_tensor = torch_state[pytorch_name]

            if tensor_name == "weight" and "token_embedding" not in tensor_path:
                pytorch_tensor = pytorch_tensor.t()

            new_tensor = jnp.array(pytorch_tensor)
        else:
            print(pytorch_name)
            raise Exception("not implemented")

        assert new_tensor.shape == value.shape, f"shape[0]={new_tensor.shape}, shape[1]={value.shape}"
        return new_tensor.astype("float32")

    return name_iter(jax_params, "", process_node)


def load(name: str, device: str = "cpu") -> tuple[CLIP, Callable[[np.ndarray[float]], np.ndarray[float]]]:
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : jax.Device
        The device to put the loaded model

    Returns
    -------
    model : flax.nnx.Model
        The CLIP model

    preprocess : Callable[[PIL.Image], jax.Array]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """

    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    
    with open(model_path, 'rb') as opened_file:
        try:
            state_dict = torch.jit.load(opened_file, map_location="cpu").eval().state_dict()
        except RuntimeError:
            state_dict = torch.load(opened_file, map_location="cpu").eval().state_dict()

    jax_device = jax.devices(device)[0]

    clip_params = get_params(state_dict)
    clip = CLIP(**clip_params)
    jax_state = jax.device_put(nnx.state(clip), jax_device)
    jax_state = jax_state.to_pure_dict()
    
    # load weights
    new_jax_state = convert_params(state_dict, jax_state)
    for key in new_jax_state.keys():
        jax_state[key] = new_jax_state[key]
    
    jax_state = nnx.State(jax_state)
    clip_graphdef = nnx.graphdef(clip)
    clip: CLIP = nnx.merge(clip_graphdef, jax_state)

    return clip, _transform(clip_params["image_resolution"])


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> jax.Array:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    result = jnp.zeros((len(all_tokens), context_length), dtype=int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        elif len(tokens) < context_length:
            padding_len = context_length - len(tokens)
            tokens.extend([eot_token for _ in range(padding_len)])

        result.at[i].set(jnp.array(tokens))

    return result
