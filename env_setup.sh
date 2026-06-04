conda create -n dit_feature python=3.8
conda activate dit_feature

pip install -q tensorflow

pip install tfds-nightly==4.4.0.dev202201080107
pip install opencv-python
pip install tensorflow-addons
pip install mock


conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install fvcore

conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor
conda install -c iopath iopath


# for transformers
pip install timm==1.0.11
pip install ml-collections

# Optional: for slurm jobs
pip install submitit -U
pip install slurm_gpustat

pip install diffusers==0.29.2 transformers==4.46.3 #controlnet_aux
pip install accelerate
pip install einops
