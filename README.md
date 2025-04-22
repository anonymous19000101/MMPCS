# MMPCS

本仓库包含了MMPCS模型的代码和对比的其中三个模型的代码

提供MMPCS的环境

```bash
conda cteate -n mmpcs python=3.10
conda activate mmpcs

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_cluster-1.6.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.17%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt20cu118-cp310-cp310-linux_x86_64.whl

pip install  torch_cluster-1.6.1+pt20cu118-cp310-cp310-linux_x86_64.whl 
pip install  torch_scatter-2.1.1+pt20cu118-cp310-cp310-linux_x86_64.whl
pip install torch_sparse-0.6.17+pt20cu118-cp310-cp310-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt20cu118-cp310-cp310-linux_x86_64.whl 

pip install rdkit==2022.3.5
pip install torch_geometric==2.3.1
pip install transformers==4.33.2
pip install numpy==1.26.1
pip install tabulate==0.9.0
```

## 运行代码

```bash
python main.py

python main_finetune_classidier.py
python main_finetune_regression_tf.py
```