# A-SelecT

This repository provides an implementation of the paper “A-SelecT: Automatic Timestep Selection for Diffusion Transformer Representation Learning”.

## Paper

Liu, C., "A-SelecT: Automatic Timestep Selection for Diffusion Transformer Representation Learning", CVPR 2026. Paper (open access): https://openaccess.thecvf.com/content/CVPR2026F/papers/Liu_A-SelecT_Automatic_Timestep_Selection_for_Diffusion_Transformer_Representation_Learning_CVPRF_2026_paper.pdf

## Quickstart

1. Create the Python environment and install dependencies:

```bash
./env_setup.sh
```

2. To use the customized diffusers implementation provided in this repository, overwrite the corresponding files in your local diffusers package with the versions located in src/diffusers/models.

3. Train a model (example for CUB):

```bash
cd DiT_feature
python train.py \
	--config-file configs/configs_dit/cub.yaml \
	DATA.DATAPATH /datapath/CUB_200_2011 \
	DATA.BATCH_SIZE 8 \
	MODEL.PROMPT.NUM_TOKENS 0 \
	MODEL.PROMPT.DROPOUT 0.0 \
	MODEL.PROMPT_HEAD_PATH "" \
	MODEL.MLP_NUM 0 \
	SOLVER.TOTAL_EPOCH 28 \
	MODEL.T_LIST "[950]" \
	MODEL.FEATURES_LAYER_LIST "[9]" \
	MODEL.FEATURES_TYPE_LIST "['query']" \
	DATA.CROPSIZE 512 \
	MODEL.FUSION_PRE_LAYER True \
	MODEL.FUSION_PRE_LAYER_TYPE attn_dim \
	MODEL.FUSION_ARC "Use_CLS_Token:True:800,Insert_CLS_Token,Attention:800:8:4:2,Extract_CLS_Token" \
	DBG True \
	MODEL.FUSION_TYPE attention \
	SEED 0 \
	MODEL.TYPE prompted-dit
```

Adjust command-line flags and config values to match your experiment needs.

## Important Notes
- Hugging Face access: if the workflow requires specific SD3.5 checkpoints, make sure your account has the required model access and your `huggingface-cli` is authenticated.
- `diffusers` compatibility: this project includes modified `diffusers` components under `src/diffusers/models`. If you encounter API mismatches, replace the corresponding files in your installed `diffusers` package with these versions.
- Timestep convention: this implementation uses a timestep convention different from some papers. When comparing timesteps with the reference (paper) convention, compute `1000 - t` where `t` is the paper timestep.
