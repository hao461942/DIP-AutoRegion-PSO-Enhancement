# 自动区域划分与 PSO 自适应图像增强

数字图像处理课程实验：冻结的 DINOv2 局部特征与 SLIC 辅助分区，通过区域统计确定处理需求，使用 PSO 选择 Gamma、CLAHE 等经典算法的参数，并平滑融合亮度增益。附加选择性暗通道处理和天空亮度协调，减轻逆光树干与枝叶上的灰白光幕。

![原图与最终结果](技术报告/images/03_result.png)

完整方法、公式、超参数、实验过程和结果分析见[技术报告](技术报告/技术报告.md)。

## 安装与运行

已验证环境：Windows、Python 3.12.13、PyTorch 2.5.1+cu118、RTX 3060 Laptop GPU 6 GB。建议使用独立环境，模型缓存保存在项目目录，不训练模型。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --check
.\.venv\Scripts\python.exe main.py
```

首次检查会从 Hugging Face 下载 `facebook/dinov2-small`，权重版本固定为 `ed25f3a31f01632728cabb09d1542f84ab7b0056`。网络需能访问 Hugging Face。输入为项目根目录 `input.jpg`，输出至 `results/`。程序保留整幅图像，输出与原图分辨率一致。

`requirements.txt` 固定主要依赖；`environment-lock.txt` 记录实际环境的完整包版本。复现以 Python 3.12 为准。更换输入时替换 `input.jpg`，然后执行完整运行命令。

## 文件与参数

- `main.py`：特征提取、自动分区、需求判断、PSO、增益融合和评价；阈值与优化设置集中于 `CFG`，算法范围见 `BOUNDS`。
- `glare.py`：FADE 统计特征、光幕候选、选择性暗通道处理及自动参考天空协调；参数集中于 `CONFIG`。
- `models_cache/FADE/`：原 FADE 发布包的两组小型统计参数及版权说明，不含增强网络。
- `技术报告/`：Markdown 报告与全部配图，可单独复制阅读。
- `results/`：区域信息、参数、收敛数据、评价表和最终图像预览。

PSO 为 Global-best，12 粒子、15 次迭代，种子42，惯性0.7，学习因子1.5/1.5。暗部搜索 Gamma [0.45,1] 与 CLAHE [1,3]；高亮搜索 Gamma [1,2.5]。曝光目标根据原图局部亮度确定，并约束过度提亮、裁剪和对比度改变。光幕与天空步骤采用固定参数，没有额外 PSO。

## 本图结果

原图为5120×3840，分析尺度为1008×756，自动得到16个连通区域：9个过暗、2个高亮、5个正常。本图未检出可靠的模糊区域，因此没有启用锐化。平均亮度从0.31720变为0.34320，局部对比度从0.06592变为0.06739。

纹理块模糊指标从0.27268变为0.27456；部分树干光晕仍存在。原图右侧天空大面积近白，压暗后无法恢复缺失纹理。结果仅来自这一张图像，不作跨场景性能推断。详细原始数值见 CSV；报告中的局部内容名称仅用于说明结果，不参与自动分区。

仓库提供的 `results/final.jpg` 为最终 PNG 的高质量预览；程序正式输出 `final.png` 和 `D.png`。模型权重、虚拟环境、下载缓存、旧版本和大型中间数组不随仓库发布。完整重跑会重建所需中间文件，因此复用参数 `--reuse-segmentation`、`--glare-only` 仅适用于本机已经完成一次完整实验的目录。历史实验修改及阶段耗时见报告，仓库数据保留最终参数来源。

## 来源

DINOv2 使用 [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) 冻结权重。FADE Python 实现依据 [UT Austin LIVE FADE 软件发布包](https://live.ece.utexas.edu/research/fog/FADE_release.zip) 适配，保留其完整版权声明及统计参数；使用本项目进行研究时请引用报告所列 FADE 文献。本项目未将 Python 适配结果与 MATLAB 实现作逐值一致性验证。

## 分区细化

DINOv2输出去除CLS后为3888×384，PCA降至16维，再恢复54×72网格并插值。对SLIC超像素内的深度与亮度、对比度、色度特征取均值，标准化后按面积加权K-means分为5类；连通拆分、大区域二分及相邻小区域合并后得到16个区域。SLIC目标1400、紧致度8、预平滑0.6；小区域门槛0.5%、大区域分裂门槛25%、区域上限16。

完成一次完整实验后，可以运行 `.\.venv\Scripts\python.exe main.py --reuse-features` 复用DINO特征并重建分区和全部增强结果。该选项要求当前输入及分析尺寸与缓存特征一致。分区步骤和张量尺寸的详细推导见技术报告第2.2节。
