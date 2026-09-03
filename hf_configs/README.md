# Bundled Hugging Face configs (no weights)

Config / tokenizer files only, for building HF backbones and CLIP text towers with
``use_pretrain=False`` while loading weights from a checkpoint later.

Used by OPUS DINOv3 ConvNeXt-B:

```
hf_configs/
  openai/clip-vit-base-patch32/
  facebook/dinov3-convnext-base-pretrain-lvd1689m/
```

Resolved automatically by ``opus.utils.hf_hub_local.resolve_hf_hub_path``.
Override root via ``MMDET_HF_CONFIG_ROOT``.
