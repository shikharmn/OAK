eval "$(conda shell.bash hook)"

yes | conda create -n oak python=3.9
conda activate oak

yes | conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia             # For most NVidia GPUs
# yes | conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia             # For H100s
yes | conda install ipykernel jupyter ipywidgets
python -m ipykernel install --user --name=oak

pip install numba pandas Cython hnswlib accelerate
pip install hydra-core --upgrade
pip install transformers sentence_transformers

pip install onnxruntime==1.14
pip install onnxruntime-gpu==1.14
pip install git+https://github.com/kunaldahiya/pyxclib.git

pip install wandb
pip install colorama