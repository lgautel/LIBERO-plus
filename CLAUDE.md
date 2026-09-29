# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LIBERO-Plus is a robustness benchmark for Vision-Language-Action (VLA) models, extending the original LIBERO framework. It evaluates VLA models across 7 perturbation dimensions (camera viewpoints, robot initial states, language instructions, lighting, background textures, sensor noise, object layout) with 10,030 tasks. It is a drop-in replacement for LIBERO — install with `pip install -e .` and existing LIBERO code works unchanged.

## Installation

```bash
pip install -e .
apt install libexpat1 libfontconfig1-dev libpython3-stdlib
apt-get install libmagickwand-dev
pip install -r extra_requirements.txt
```

Assets must be downloaded from HuggingFace and extracted to `libero/libero/assets/`.

## Commands

### Training
```bash
python libero/lifelong/main.py \
    benchmark_name=LIBERO_SPATIAL \
    policy=bc_transformer_policy \
    lifelong=single_task \
    seed=10000
```
Override any Hydra config on the command line. Config root: `libero/configs/config.yaml`.

### Evaluation
```bash
python libero/lifelong/evaluate.py
```
For LIBERO-Plus evaluation, set `num_trials_per_task` to 1 (original LIBERO uses 50).

### Console entry points (registered in setup.py)
- `lifelong.main` — training
- `lifelong.eval` — evaluation

## Architecture

### Two-level package layout: `libero/` (top) and `libero/libero/` (core)

**`libero/libero/`** — simulation environment and benchmark definitions:
- `benchmark/` — Benchmark classes (`LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_10`, `LIBERO_90`, `LIBERO_100`, `LIBERO_MIX`), registered via `@register_benchmark` decorator. Task-to-perturbation mapping in `task_classification.json`.
- `envs/` — MuJoCo/robosuite environments. `problems/` defines per-scene manipulation tasks (kitchen, living room, study, etc.). `arenas/` defines scene layouts. `robots/` defines Panda arm variants. `env_wrapper.py` contains `OffScreenRenderEnv`/`SegmentationRenderEnv` and sensor-noise corruption functions (motion blur, Gaussian blur, etc.).
- `bddl_files/` — BDDL task specifications organized by suite (`libero_spatial/`, `libero_object/`, `libero_goal/`, `libero_10/`, `libero_90/`, `libero_mix/`).
- `init_files/` — serialized initial states for each task (`.pruned_init` files loaded via `torch.load`).
- `utils/` — helpers for BDDL generation, datasets, video recording, object manipulation.
- `__init__.py` — manages path configuration via `~/.libero/config.yaml`. On first import, prompts for dataset path.

**`libero/lifelong/`** — training and evaluation:
- `algos/` — lifelong/continual learning algorithms: `SingleTask`, `Multitask`, `ER`, `AGEM`, `EWC`, `PackNet`. All inherit from `Sequential` base class.
- `models/` — policy networks: `BCRNNPolicy`, `BCTransformerPolicy`, `BCViLTPolicy`. Registered via `get_policy_class()`.
- `main.py` — Hydra-based training entry point.
- `evaluate.py` — evaluation loop with multiprocessing support.
- `datasets.py` — `SequenceVLDataset`, `GroupedTaskDataset` for loading HDF5 demonstrations.

**`libero/configs/`** — Hydra YAML configs for `data`, `eval`, `lifelong`, `policy`, `train`.

### Perturbation naming conventions in task/file names
Suffixes encode which perturbation is applied: `_view_` (camera), `_language_` (instruction rewriting), `_table_`/`_tb_` (background texture), `_light_` (lighting), `_add_`/`_level` (object layout). The `Benchmark.get_task_init_states()` method uses these suffixes to resolve the correct init-state file.


# 介绍

- 这是论文[LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models](https://arxiv.org/abs/2510.13626)的代码库, 论文的html版在 https://arxiv.org/html/2510.13626v3 . 
- 论文的项目主页在 https://sylvestf.github.io/LIBERO-plus , 
- 论文的GitHUb在 https://github.com/sylvestf/LIBERO-plus ,
- 论文的数据在 https://huggingface.co/datasets/Sylvest/libero_plus_rlds , https://huggingface.co/datasets/Sylvest/libero_plus_lerobot 或 https://huggingface.co/datasets/Sylvest/libero_plus_data_4suite/tree/main 
- 论文涉及的 3D 仿真资产在 https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main  


# 设计/方案/分析/解释和写文档的注意点

* 图表用mermaid, 数学相关的用LaTex, 必要时可以用py脚本画一些更能帮助读者理解的图片(图片中的文字用英文). 这些脚本和图一般放在与生成的文档同目录的`asset`子文件夹中.
* 如果在公式和内容中用到了数学符号或代号, 请在该公式或内容的附近对该符号给予解释.
* 分析,解析和撰写文档时, 可以参考论文或代码库的官网, 官方文档, GItHUb, 参考github中的issues, 代码和pull requests, 也可参考网上其它可信来源的相关文章, 但参考内容要列出, 所生产的文档中若有与被参考对象相关的内容也要指出内容的出处. 
* 分析要深入仔细, 既要包括纵向分析(算法或方法的由来与演进历史, 以及在该算法或方法的基础上又演进和优化出了些什么解决类似问题的方法, 新老方法各有什么优缺点, 各适合应用到什么场景), 纵向分析(同时期同类算法的对比分析, 不同算法或方法各有什么优缺点, 各适合应用到什么场景), 和 消融分析(算法或方法中哪些点是在benchmark实验或实践中被证明有效的, 哪些点相对来说更有效, 哪些没那么有效).
* 记得深入分析模型或方法的输入,输出,在输入输出间做了些什么处理. 当然, 各组成模块的输入输出以及中间的处理也要分析. 为了训这个模型用了什么数据集和任务, 训出来后能做什么任务, 训练和推理时的输入输出数据格式大概长什么样.
* 系统或程序的设计要包括静态架构(组件图,类图,组件和类的职责与关系等等)和动态架构(数据流图,序列图,工作流图,不同场景下的各组件或类的调用与协调图.如果是算法还会涉及forward阶段的数据流,模型组件间的调用,以及backwawrd阶段的数据流,gradient流,哪些权重冻结哪些会被更新,和模型组件间的调用等等).
* 如果是设计与实施落地相关的文档, 要遵守这些设计原则: 
    - 扩展由于修改, 尽量通过各种设计模式来扩展新模块新功能, 而不是通过修改原来的代码得到新特性; 
    - 尽量复用原有代码, 若不能复用要给出理由; 会随着软硬件环境, 机器人, 底层框架或者云上环境变化而变化的点, 要抽象出来, 作为关键配置点, 最好不同的软硬件环境可对应一个配置文件, 并对配置项和配置文件做详细说明; 
    - 会随着实验的不同, 数据准备, 训练, 评估的不同而变化的点, 也要抽象出来, 作为配置关键点, 最后不同的实验可对应一个配置文件, 并对配置项和配置文件做详细说明.
* 如果是设计与实施落地相关的文档, 要给出测试方案, 验收方案, 以及相关的代码和脚本, 并对方案和代码/脚本的输入输出, 测试前提, 验收条件, 覆盖和没覆盖的分支等相关细节进行详细解释.按设计与实施落地文档执行时,除了要把所有工作完成外, 还要把所有测试和评估都通过了才算成功.
* 代码还是以该代码库的本地代码为准, 但可用参考网上GitHub的issues, commits, pull requests等.
* 解释要深入浅出, 图文并茂, 可以举一些易于理解的例子帮助说明, 对关键的逻辑也要进行深入的代码解读, 要用严谨的科普论文的风格.