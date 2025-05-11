# MMPCS

本仓库包含了MMPCS模型的代码和对比的其中三个模型的代码

首先默认环境为 Ubuntu 22.04 LST

## 环境

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

### 数据文件

链接：https://pan.baidu.com/s/1hUuk3HtASd_LvfiefzsdHA 
提取码：yqfv 
--来自百度网盘超级会员V1的分享

包含了预训练数据集、微调数据集，以及roberta的预训练模型

### 预训练

```bash
python main.py
```

### 微调

预训练结束会在当前目录下生成一个文件夹，记住这个文件夹的名字，然后作为finetune时候读预训练模型的文件夹

```bash
# 分类任务微调，以 Estrogen，多卡 为例
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 main_finetune_classifier.py --dataset Estrogen --is_multi_gpu True --gpu cuda
```

```bash
# 回归任务微调，以 ESOL，单卡 为例
python main_finetune_regression.py --dataset ESOL --gpu cuda:1 
```

如果想要自己增加自定的数据集，那么需要在Dataset_info.py 文件中进行配置

### 增加自定义微调数据集

#### 准备你的数据集文件

CSV 格式： 确保你的数据集是 CSV 格式。
SMILES 列： 文件中必须有一列包含分子的 SMILES 字符串。

标签列：
分类任务： 文件中应包含一列或多列作为分类任务的标签。这些标签通常是整数（如 0 或 1）。如果一个分子有多个任务，则每个任务对应一列。
回归任务： 文件中应包含一列作为回归任务的目标值（数值型）。
文件位置： 将你的 CSV 文件（例如 my_new_dataset.csv）放入 Config.FINETUNE_DATASET_BASEPATH 指定的文件夹中。

#### 修改 DatasetInfo 类的 load_info 方法

打开包含 DatasetInfo 类的 Python 文件。找到 load_info(self) 方法。你需要在这个方法中的 if/elif 链中添加一个新的 elif 块来定义你的数据集。
假设你的新数据集名为 MyNewDataset，它是一个分类任务，SMILES 在名为 molecule_smiles 的列，标签有3个任务，分别在 task1_label, task2_label, task3_label 列，CSV文件名为 my_new_dataset.csv，你希望使用 Scaffold 分割，并用 ROC_AUC 评估。
在 load_info 方法的末尾，else: raise ValueError("数据集不存在") 之前，添加如下代码块：

```python
# ... 其他数据集的 elif 块 ...

        # 已设计好
        elif self.dataset_name == "BACE":
            # 只留下 mol 和 Class 其他全要去掉.
            self.type = "classification"
            self.tasks_num = 1
            self.filename = "bace.csv"
            self.smiles_col = "mol" # 注意这里SMILES列名是 "mol"

            self.split_type = "Scaffold"
            self.eval_metrics = "ROC_AUC"

        # -------------------------------------------------------------
        # 在这里添加你的新数据集配置
        elif self.dataset_name == "MyNewDataset":  # <--- 1. 你的数据集名称
            self.type = "classification"          # <--- 2. 任务类型: "classification" 或 "regression"
            self.tasks_num = 3                    # <--- 3. (分类任务) 任务数量
            self.filename = "my_new_dataset.csv"  # <--- 4. 你的CSV文件名
            self.smiles_col = "molecule_smiles"   # <--- 5. SMILES字符串所在的列名

            # self.label_col = "your_target_value" # <--- 6. (回归任务) 标签所在的列名
                                                #      分类任务不需要设置这个

            self.split_type = "Scaffold"          # <--- 7. 数据分割方式 (例如 "Scaffold", "Rondom")
            self.eval_metrics = "ROC_AUC"         # <--- 8. 评估指标 (例如 "ROC_AUC", "RMSE")
        # -------------------------------------------------------------

        # 以下五个是回归任务
        elif self.dataset_name == "ESOL":
# ... (代码继续) ...
        else:
            # 报错
            raise ValueError("数据集不存在")
```

配置项说明：
self.dataset_name == "MyNewDataset": 将 "MyNewDataset" 替换为你的数据集的唯一标识符。
self.type:
设置为 "classification" 如果是分类任务。
设置为 "regression" 如果是回归任务。
self.tasks_num (仅限分类任务):
如果你的分类数据集只有一个标签列（单任务分类），设置为 1。
如果你的分类数据集有多个标签列（多任务分类），设置为任务的数量（例如，3个标签列，则为 3）。
self.filename: 你的数据集的 CSV 文件名 (例如 "my_new_dataset.csv")。
self.smiles_col: CSV 文件中包含 SMILES 字符串的列的名称 (例如 "molecule_smiles")。
self.label_col (仅限回归任务):
CSV 文件中包含回归目标值的列的名称 (例如 "experimental_value")。
对于分类任务，不要设置此项，DatasetLoadHelper 会从多个任务列中提取标签。
self.split_type: 数据集划分的策略，根据你的需求选择 (例如 "Scaffold", "Rondom")。
self.eval_metrics: 用于评估模型性能的指标。
分类任务常用: "ROC_AUC。metric.py 文件中定义了一些指标的计算方法，如果需要使用自定义指标，请在metric.py 中添加相应的计算函数。
回归任务常用: "RMSE" 。
对于回归任务的示例：
假设你有一个名为 MySolubility 的回归数据集，SMILES 在 smiles_string 列，溶解度值在 logS 列，文件名为 solubility_data.csv。

```python
# ... 其他数据集的 elif 块 ...
        elif self.dataset_name == "MySolubility":
            self.type = "regression"
            self.filename = "solubility_data.csv"
            self.smiles_col = "smiles_string"
            self.label_col = "logS"  # 回归任务需要指定标签列

            self.split_type = "Rondom"
            self.eval_metrics = "RMSE"
        # ...
```

#### 修改 DatasetLoadHelper 类

这一步主要针对分类任务，因为它们通常涉及从多个列中提取标签，或者需要从原始CSV中去除SMILES列和一些非标签的ID列。回归任务通常只需要指定单一的 label_col，其加载逻辑在 DatasetLoadHelper 中已经通用化处理了。
在 load_dataset 方法中添加条件分支：
打开 DatasetLoadHelper 类，找到 load_dataset(self) 方法。在 elif self.datasetinfo.type == "classification": 块内部的 if/elif 链中，为你的新数据集添加一个条件。

```python
# class DatasetLoadHelper:
# ...
#     def load_dataset(self):
#         if self.datasetinfo.type == "regression":
#             self.tasks_labels_list = pd.read_csv(self.file_path, index_col=None)[self.datasetinfo.label_col].tolist()
#         elif self.datasetinfo.type == "classification":
#             if self.datasetinfo.dataset_name == "BBBP":
#                 return self.load_BBBP()
#             # ... 其他已存在的分类数据集 ...
#             elif self.datasetinfo.dataset_name == "MetStab":
#                 return self.load_MetStab()
#             # -------------------------------------------------------------
#             # 在这里为你的新分类数据集添加调用
#             elif self.datasetinfo.dataset_name == "MyNewDataset": # <--- 你的数据集名称
#                 return self.load_MyNewDataset()                 # <--- 调用新的加载方法
#             # -------------------------------------------------------------
#             else:
#                 raise ValueError("分类数据集的特定加载方法不存在或数据集名称错误") # 修改了错误信息
# ...
```

创建新的加载方法 load_MyNewDataset(self)：
在 DatasetLoadHelper 类中，添加一个新的方法来具体处理你的新分类数据集的标签加载。这个方法的名称应该与上一步中调用的名称一致（例如 load_MyNewDataset）。
核心任务：
读取 CSV 文件。
移除不作为标签的列，特别是 SMILES 列和任何其他ID或非特征列。
剩下的列应该都是你的任务标签列。
调用 self.get_task_info(df)，其中 df 是只包含标签列的 DataFrame。

```python
# class DatasetLoadHelper:
# ...
# (其他 load_ 方法) ...

def load_MetStab(self):
    df = pd.read_csv(self.file_path, index_col=None).drop(["smiles"], axis=1)
    tasks_num = df.shape[1]
    print("tasks_num:", tasks_num)
    self.get_task_info(df)

# -------------------------------------------------------------
# 在这里实现你的新数据集的加载方法 (主要针对分类任务)
def load_MyNewDataset(self):
    # 读取CSV文件
    # 假设你的CSV有 'molecule_smiles', 'id_col', 'task1_label', 'task2_label', 'task3_label' 列
    # 我们需要去掉 'molecule_smiles' (因为它是SMILES) 和 'id_col' (假设它是一个无关ID)
    # 剩下的 'task1_label', 'task2_label', 'task3_label' 就是标签
    df = pd.read_csv(self.file_path, index_col=None)

    # 获取SMILES列名，以便从DataFrame中移除
    smiles_column_name = self.datasetinfo.smiles_col

    # 定义需要移除的非标签列，至少要包含SMILES列
    # 如果还有其他如 'id', 'name' 等非标签列，也加入到这个列表中
    columns_to_drop = [smiles_column_name]
    # 示例：如果你的CSV还有一个叫 'molecule_id' 的非标签列，则：
    # columns_to_drop.append('molecule_id')

    df_labels_only = df.drop(columns=columns_to_drop, axis=1)

    # 验证DataFrame中剩余的列数是否与 DatasetInfo 中配置的 tasks_num 一致
    # 这是可选的，但有助于调试
    actual_tasks_num = df_labels_only.shape[1]
    configured_tasks_num = self.datasetinfo.tasks_num
    if actual_tasks_num != configured_tasks_num:
        print(f"警告: MyNewDataset 中检测到的任务列数 ({actual_tasks_num}) 与配置的任务数 ({configured_tasks_num}) 不符。")
        print(f"DataFrame 中的列为: {df_labels_only.columns.tolist()}")
        # 你可以选择在这里抛出错误，或者继续（但可能会导致后续问题）
        # raise ValueError(f"任务数量不匹配! 配置: {configured_tasks_num}, 实际: {actual_tasks_num}")


    print(f"MyNewDataset - tasks_num: {actual_tasks_num}")
    self.get_task_info(df_labels_only) # 传入只包含标签列的DataFrame
# -------------------------------------------------------------
```

重要：
columns_to_drop: 确保这个列表包含了你的 SMILES 列名 (从 self.datasetinfo.smiles_col 获取) 以及 CSV 文件中任何其他不属于标签的列。
df_labels_only: 传递给 self.get_task_info() 的 DataFrame (df_labels_only) 必须只包含标签列。get_task_info 会将这个 DataFrame 的每一列都视为一个独立的任务。
如果你的回归任务有特殊的列处理（例如，需要从多个列计算得到最终的label，或者需要重命名列），你也可能需要类似的自定义加载方法。但通常，回归任务的标签是单一列，不需要自定义加载器。


### 消融及参数实验

#### 不进行预训练

```bash
# 以 SIDER 数据集为例
python main_finetune_classifier.py --dataset SIDER --gpu cuda:5 --model_dir no_pretrain --pretrained False
```

其中 `pretrained` 参数为 False 即可，``model_dir` 为保存模型的文件夹名

#### 只是用 Transformer 分支进行推理

```bash
python main_finetune_classifier_tf.py --dataset BBBP --gpu cuda:1
```


#### 只是用 GNN 分支进行推理

```bash
python main_finetune_classifier_gnn.py --dataset BBBP --gpu cuda:2
```

#### 不使用 CoV 损失函数

```bash
python main_finetune_classifier_wo_cov.py --dataset BBBP --gpu cuda:7
```

#### 不使用 align 损失函数

```bash
python main_finetune_classifier_wo_align.py --dataset BBBP --gpu cuda:6
```

#### GNN 层数参数实验

```bash
# 下面是两层的GNN的情况
python main.py --gnn_layer 2 --model_dir gnn2
python main_finetune_classifier.py --dataset SIDER --gpu cuda:0 --model_dir gnn2 --gnn_layer 2
python main_finetune_regression.py --dataset ESOL --gpu cuda:0 --model_dir gnn2 --gnn_layer 2
```

#### alpha beta 参数实验

需要重新预训练

```bash
python main.py --model_dir beta2 --beta 2 --batch_size 64 --device cuda:1
python main_finetune_classifier.py --dataset BBBP --gpu cuda:1 --model_dir beta2 --beta 2
```