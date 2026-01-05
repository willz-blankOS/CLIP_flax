from PIL import Image

import jax
import jax.numpy as jnp
import numpy as np

import clip
import clip_flax

def test_model(model_name):
    jax_clip, jax_preprocess = clip_flax.load(model_name)

    jax_image = jnp.array(np.expand_dims(jax_preprocess(Image.open("CLIP.png")), 0))
    jax_image = jax_image.transpose((0, 2, 3, 1))
    jax_text = clip_flax.tokenize(["a diagram", "a dog", "a cat"])

    jax_image_embed = jax_clip.encode_image(jax_image)
    jax_text_embed = jax_clip.encode_text(jax_text)
    
    pytorch_clip, pyt_preprocess = clip.load(model_name, "cpu")

    pyt_image = pyt_preprocess(Image.open("CLIP.png")).unsqueeze(0).to("cpu")
    pyt_text = clip.tokenize(["a diagram", "a dog", "a cat"])

    pyt_image_embed = pytorch_clip.encode_image(pyt_image)
    pyt_text_embed = pytorch_clip.encode_text(pyt_text)

    pyt_image = pyt_image.cpu().detach().numpy()
    jax_image = np.array(jax_image)

    pyt_text_embed = pyt_text_embed.cpu().detach().numpy()
    print(pyt_image_embed)
    jax_text_embed = np.array(jax_text_embed)

    assert np.allclose(jax_image_embed, pyt_image_embed, atol=0.01, rtol=0.01)
    assert np.allclose(jax_text_embed, pyt_text_embed, atol=0.01, rtol=0.01)

    print(f"{model_name}: done!")

test_model("ViT-B/32")
test_model("ViT-B/16")
test_model("ViT-L/14")