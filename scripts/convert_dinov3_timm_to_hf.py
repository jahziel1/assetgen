"""Convierte timm/vit_large_patch16_dinov3_qkvb.lvd1689m a transformers.DINOv3ViTModel y verifica."""
import os, sys, numpy as np, torch, torch.nn.functional as F, timm
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel
OUT = sys.argv[1] if len(sys.argv) > 1 else '/root/dinov3-vitl16-hf'
IMG = sys.argv[2] if len(sys.argv) > 2 else None
tm = timm.create_model('vit_large_patch16_dinov3_qkvb.lvd1689m', pretrained=True).eval()
sd = tm.state_dict()
cfg = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
    num_register_tokens=4, patch_size=16, rope_theta=100.0, layerscale_value=1.0,
    query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True, use_gated_mlp=False)
hf = DINOv3ViTModel(cfg).eval()
hsd = hf.state_dict()
pre = 'model.' if any(k.startswith('model.layer.') for k in hsd) else ''
new = {}
def put(k, v):
    new[k] = v.clone()
new['embeddings.cls_token'] = sd['cls_token'].clone()
new['embeddings.register_tokens'] = sd['reg_token'].clone()
new['embeddings.patch_embeddings.weight'] = sd['patch_embed.proj.weight'].clone()
new['embeddings.patch_embeddings.bias'] = sd['patch_embed.proj.bias'].clone()
for i in range(24):
    b, l = f'blocks.{i}.', f'{pre}layer.{i}.'
    q, k, v = sd[b+'attn.qkv.weight'].chunk(3, 0)
    put(l+'attention.q_proj.weight', q); put(l+'attention.k_proj.weight', k); put(l+'attention.v_proj.weight', v)
    put(l+'attention.q_proj.bias', sd[b+'attn.q_bias']); put(l+'attention.v_proj.bias', sd[b+'attn.v_bias'])
    put(l+'attention.o_proj.weight', sd[b+'attn.proj.weight']); put(l+'attention.o_proj.bias', sd[b+'attn.proj.bias'])
    put(l+'layer_scale1.lambda1', sd[b+'gamma_1']); put(l+'layer_scale2.lambda1', sd[b+'gamma_2'])
    for n in ('norm1', 'norm2'):
        put(l+n+'.weight', sd[b+n+'.weight']); put(l+n+'.bias', sd[b+n+'.bias'])
    put(l+'mlp.up_proj.weight', sd[b+'mlp.fc1.weight']); put(l+'mlp.up_proj.bias', sd[b+'mlp.fc1.bias'])
    put(l+'mlp.down_proj.weight', sd[b+'mlp.fc2.weight']); put(l+'mlp.down_proj.bias', sd[b+'mlp.fc2.bias'])
new['norm.weight'] = sd['norm.weight'].clone(); new['norm.bias'] = sd['norm.bias'].clone()
missing, unexpected = hf.load_state_dict(new, strict=False)
print('missing', missing, 'unexpected', unexpected)   # esperado: solo embeddings.mask_token (y buffers rope)
hf.save_pretrained(OUT)
hf = DINOv3ViTModel.from_pretrained(OUT).eval()       # recarga desde disco

if IMG and os.path.exists(IMG):
    im = Image.open(IMG).convert('RGB').resize((512, 512), Image.LANCZOS)
else:   # sin imagen: patron sintetico (la verificacion numerica sigue siendo valida)
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8))
x = torch.from_numpy(np.array(im).astype(np.float32)/255).permute(2, 0, 1)[None]
x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
def cmp(name, a, b):
    a, b = a.flatten().double(), b.flatten().double()
    print(f'{name}: cosine={F.cosine_similarity(a, b, dim=0).item():.6f} max_abs_diff={(a-b).abs().max().item():.2e} mean_abs_diff={(a-b).abs().mean().item():.2e} (|ref| max={b.abs().max().item():.1f})')
with torch.no_grad():
    ref = tm.forward_features(x)                       # timm: con norm final
    cmp('A) HF last_hidden_state vs timm forward_features', hf(pixel_values=x).last_hidden_state, ref)
    # B) camino que usa TRELLIS (extract_features: capas + layer_norm sin afin)
    h = hf.embeddings(x, bool_masked_pos=None); pe = hf.rope_embeddings(x)
    for lm in hf.layer: h = lm(h, position_embeddings=pe)
    mine = F.layer_norm(h, h.shape[-1:])
    tm.norm = torch.nn.Identity()
    cmp('B) TRELLIS extract_features vs timm (sin norm final)+layer_norm', mine, F.layer_norm(tm.forward_features(x), (1024,)))
