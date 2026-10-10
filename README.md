# ASCL-FS

采用主算法的 Case A参数。保留稳定双评分筛选与渐进初始化、交叉引导动态阈值、鲁棒精英档案三个核心模块。

## 运行

用 MATLAB 打开本文件夹并设为当前文件夹，打开 'main.m'后点击“运行”，或在命令窗口输入：

```matlab
main

运行环境：MATLAB R2022b 和 Statistics and Machine Learning Toolbox。

'DataSet' 已包含 Case A 结果目录对应的 21 个数据集，每个 MAT 文件包含 `feat` 和 `label`。所有输入、输出路径均相对于本项目，可整体移动。

## 文件

| 文件 | 用途 |
| --- | --- |
| `main.m` | 数据读取、嵌套交叉验证、结果保存 |
| `parameters.m` | 实验及算法参数 |
| `gASCLFS.m` | ASCL-FS 核心算法 |
| `jFitnessFunction.m` | 特征子集适应度 |
| `jknn.m` | 确定性的欧氏距离 kNN 分类 |
| `SU_TOOL.m` | 对称不确定性评分 |

`main.m` 是直接运行的脚本。修改顶部的 `datasetNames` 可选择实验数据，名称统一包含 `.mat` 后缀，例如：

```matlab
datasetNames = ["Colon.mat", "nci9.mat"];
```

修改 `parameters.m` 可调整实验及算法参数；修改 `main.m` 顶部的 `dataDir` 和 `resultDir` 可调整输入、输出位置。脚本按参数设置、数据集列表、路径设置、实验循环和结果保存组织；单数据集实验及读写操作放在文件末尾的局部函数中。`gASCLFS.m` 按筛选初始化、初始评估、迭代更新和结果输出组织。

## 实验设置

- 种群 `N = 100`，迭代次数 `T = 100`，分类器 `k = 1`。
- 外层 10 折评估，内层 5 折特征选择；先生成全部划分，再开始优化。
- 基础随机种子为 42，每次运行使用 `rng(seed + runIndex, 'twister')`，保留原来的随机数调用顺序。
- Case A 的初始阈值为 **0.4 / 0.6**，逐步收敛到 0.5。
- 适应度为 `0.90 × 内层汇总分类错误率 + 0.10 × 已选特征占原始特征比例`。
- 保留原数据处理和分类规则，不增加归一化或更换 MATLAB 内置分类器。

## 输出

运行时自动创建 `results` 目录。每个数据集生成逐次指标 CSV 和包含特征索引、收敛曲线、折划分、预测值及参数的 MAT，另有汇总表 `summary.csv`。`TestAcc` 是汇总全部外层测试预测得到的准确率，`TrainCV` 是外层各训练集内层验证准确率的均值。

相同输出目录中，同名数据集结果及汇总表会被更新；汇总表只包含本次运行的数据集。保留多个实验时，可修改脚本顶部的输出目录：

```matlab
resultDir = fullfile(projectDir, 'results', 'my_experiment');
```
