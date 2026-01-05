import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

RNGS = nnx.Rngs(42)

class Bottleneck(nnx.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # All conv layers have stride 1
        self.conv1 = nnx.Conv(
            in_channels, 
            out_channels, 
            1, 
            use_bias=False,
            rngs=RNGS
        )
        self.bn1 = nnx.BatchNorm(out_channels)
        
        self.conv2 = nnx.Conv(
            out_channels, 
            out_channels, 
            3, 
            padding=1,
            use_bias=False,
            rngs=RNGS
        )
        self.bn2 = nnx.BatchNorm(out_channels)

        self.conv3 = nnx.Conv(
            out_channels, 
            out_channels * self.expansion, 
            1, 
            use_bias=False,
            rngs=RNGS
        )
        self.bn3 = nnx.BatchNorm(out_channels * self.expansion)

        self.avgpool = nnx.avg_pool if stride > 1 else jnp.identity

        self.downsample = None
        self.stride = stride

        if stride > 1 or in_channels != out_channels * Bottleneck.expansion:
            self.downsample = nnx.Sequential(
                nnx.avg_pool,
                nnx.Conv(
                    in_channels, 
                    out_channels * self.expansion, 
                    1, 
                    strides=1, 
                    use_bias=False,
                    rngs=RNGS
                ),
                nnx.BatchNorm(out_channels * self.expansion)
            )

    def __call__(self, x: jax.Array):
        identity = x

        out = jax.nn.relu(self.bn1(self.conv1(x)))
        out = jax.nn.relu(self.bn2(self.conv2(out)))
        out = self.avg_pool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = jax.nn.relu(out)
        return out


class PositionalEmbedding(nnx.Module):
    def __init__(self, spacial_dim: int, embed_dim: int):
        super().__init__()
        self.positional_embedding = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(42), 
                shape=(spacial_dim ** 2 + 1, embed_dim)
            )
        )

    def __call__(self, x: jax.Array):
        return x + self.positional_embedding[:, None, :]


class AttentionPool(nnx.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = PositionalEmbedding(spacial_dim, embed_dim)
        self.num_heads = num_heads
        self.attn = nnx.MultiHeadAttention(
            num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim // num_heads,
            out_features=output_dim or embed_dim,
            rngs=nnx.Rngs(42)
        )

    def __call__(self, x: jax.Array):
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W).transpose(2, 0, 1)
        n = x.mean(axis=0, keepdims=True)
        x = jnp.concatenate([n, x], axis=0)
        x = self.positional_embedding(x).transpose(1, 0, 2)
        x = self.attn(x[:1], x, x)
        return x

class ModifiedResNet(nnx.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nnx.Conv(
            input_resolution,
            width // 2,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            use_bias=False,
            rngs=RNGS
        )
        self.bn1 = nnx.BatchNorm(width//2)

        self.conv2 = nnx.Conv(
            input_resolution,
            width,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=False,
            rngs=RNGS
        )
        self.bn2 = nnx.BatchNorm(width)

        self.conv3 = nnx.Conv(
            input_resolution,
            width,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=False,
            rngs=RNGS
        )
        self.bn3 = nnx.BatchNorm(width)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool(
            input_resolution // 32, 
            embed_dim, 
            heads, 
            output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, stride))

        return nnx.Sequential(*layers)

    def __call__(self, x: jax.Array):
        def stem(x: jax.Array):
            x = jax.nn.relu(self.bn1(self.conv1(x), True))
            x = jax.nn.relu(self.bn2(self.conv2(x), True))
            x = jax.nn.relu(self.bn3(self.conv3(x), True))
            x = nnx.avg_pool(x, 2, 2, padding="SAME")
            return x
        
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

class QuickGELU(nnx.Module):
    def __call__(self, x: jax.Array):
        return x * jax.nn.sigmoid(1.702 * x)


class MLP(nnx.Module):
    def __init__(self, 
                 d_model: int, 
                 d_inner: int, 
                 d_outer: int
                ):
        super().__init__()
        self.c_fc = nnx.Linear(
            d_model, d_inner, rngs=nnx.Rngs(42)
        )
        self.c_proj = nnx.Linear(
            d_inner, d_outer, rngs=nnx.Rngs(42)
        )
        self.gelu = QuickGELU()

    def __call__(self, x: jax.Array):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class MultiHeadAttention(nnx.Module):
    def __init__(self,
                 heads: int,
                 d_model: int,
                 w_init_scale: float,
                 attn_mask: jax.Array = None):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_head = d_model // heads
        self.w_init = nnx.initializers.variance_scaling(
            w_init_scale, mode="fan_in", distribution="truncated_normal"
        )
        self.attn_mask = attn_mask

        self.in_proj_weight = nnx.Param(
            self.w_init(nnx.Rngs(42)(), shape=(d_model * 3, d_model))
        )
        self.in_proj_bias = nnx.Param(
            nnx.initializers.zeros(key=nnx.Rngs(42), shape=(d_model * 3,))
        )
        self.out_proj = nnx.Linear(d_model, d_model, kernel_init=self.w_init, rngs=nnx.Rngs(42))

    def __call__(self, x: jax.Array):
        all_out = jnp.dot(x, self.in_proj_weight.value.transpose())
        all_out += self.in_proj_bias.value
        
        Q, K, V = jnp.array_split(all_out, 3, axis=-1)

        query_heads = self._split(Q)
        key_heads = self._split(K)
        value_heads = self._split(V)

        attention_logits = jnp.einsum("IBHD,JBHD->BHIJ", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.d_model//self.heads).astype(K.dtype)
        attention_logits = attention_logits / sqrt_key_size

        if self.attn_mask is not None:
            attention_logits += self.attn_mask

        attention_weights = jax.nn.softmax(attention_logits)
        attention = jnp.einsum("bhtT,Tbhd->tbhd", attention_weights, value_heads)
        attention_vec = jnp.reshape(attention, (*Q.shape[:2], -1))

        return self.out_proj(attention_vec)

    def _split(self, x: jax.Array):
        return x.reshape((*x.shape[:2], self.heads, self.d_head))

class ResidualAttentionBlock(nnx.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: jax.Array):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, d_model, w_init_scale=1.0)
        self.ln_1 = nnx.LayerNorm(d_model, use_scale=True, use_bias=True, rngs=nnx.Rngs(42))
        self.mlp = MLP(d_model, d_model * 4, d_model)
        self.ln_2 = nnx.LayerNorm(d_model, use_scale=True, use_bias=True, rngs=nnx.Rngs(42))

    def __call__(self, x: jax.Array):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nnx.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: jax.Array = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = [
            ResidualAttentionBlock(width, heads, attn_mask) 
            for _ in range(layers)
        ]

    def __call__(self, x: jax.Array):
        for block in self.resblocks:
            x = block(x)
        return x


class VisualTransformer(nnx.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=width,
            kernel_size=(patch_size, patch_size),
            strides=patch_size,
            use_bias=False,
            padding='SAME',
            rngs=RNGS
        )
        #print(f"Conv weight: {self.conv1.kernel.value.shape}"); exit()

        self.w_init = nnx.initializers.truncated_normal(1.0 / np.sqrt(width))

        self.class_embedding = nnx.Param(
            self.w_init(nnx.Rngs(42)(), shape=(width,))
        )
        self.positional_embedding = nnx.Param(
            self.w_init(
                nnx.Rngs(42)(), 
                shape=((input_resolution // patch_size)**2 + 1, width)
            )
        )
        self.ln_pre = nnx.LayerNorm(width, use_scale=True, use_bias=True, rngs=nnx.Rngs(42))

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nnx.LayerNorm(width, use_scale=True, use_bias=True, rngs=nnx.Rngs(42))
        self.proj = nnx.Param(
            self.w_init(nnx.Rngs(42)(), shape=(width, output_dim))
        )

    def __call__(self, x: jax.Array):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = jnp.concatenate(
            [self.class_embedding.value[None, None, :] + jnp.zeros((x.shape[0], 1, x.shape[-1])), x],
            axis=1
        )
        x = x + self.positional_embedding
        
        x = self.ln_pre(x)
        x = x.transpose((1, 0, 2))

        x = self.transformer(x)
        x = x.transpose((1, 0, 2))

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nnx.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                ):
        super().__init__()
        self.context_length = context_length
        
        vision_heads = vision_width // 64

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads, 
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nnx.Embed(vocab_size, transformer_width, rngs=nnx.Rngs(42))

        scale = transformer_width ** -0.5
        w_init = nnx.initializers.truncated_normal(1.0 / np.sqrt(scale))
        self.positional_embedding = nnx.Param(
            w_init(nnx.Rngs(42)(), shape=(self.context_length, transformer_width))
        )
        self.ln_final = nnx.LayerNorm(transformer_width, use_scale=True, use_bias=True, rngs=nnx.Rngs(42))

        self.text_projection = nnx.Param(
            w_init(nnx.Rngs(42)(), shape=(transformer_width, embed_dim))
        )
        self.logit_scale = nnx.Param(
            jnp.array(1)
        )

    def build_attention_mask(self):
        mask = jnp.zeros((self.context_length, self.context_length))
        mask += jnp.array(float("-inf"))
        mask = jnp.triu(mask, 1)
        return mask
    
    def encode_image(self, image: jax.Array):
        return self.visual(image)

    def encode_text(self, text: jax.Array):
        x = self.token_embedding(text)

        x = x + self.positional_embedding.value
        x = x.transpose((1, 0, 2))
        x = self.ln_final(x)

        x = x[jnp.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection
        return x

    def __call__(self, image: jax.Array, text: jax.Array):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features: jax.Array = (
            image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        )
        text_features: jax.Array = (
            text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
        )

        # cosine similarity as logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.transpose()
        logits_per_text = logit_scale * text_features @ image_features.transpose()

        return logits_per_image, logits_per_text


def get_params(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2]  for k in state_dict if k.startswith(f"transformer.resblocks")))

    return {
        "embed_dim": embed_dim,
        "image_resolution": image_resolution,
        "vision_layers": vision_layers,
        "vision_width": vision_width,
        "vision_patch_size": vision_patch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "transformer_width": transformer_width,
        "transformer_heads": transformer_heads,
        "transformer_layers": transformer_layers
    }
