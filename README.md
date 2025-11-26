<h1>TD-MPC2</span></h1>

This repository is based upon the official implementation of

[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://www.tdmpc2.com) by

[Nicklas Hansen](https://nicklashansen.github.io), [Hao Su](https://cseweb.ucsd.edu/~haosu)\*, [Xiaolong Wang](https://xiaolonw.github.io)\* (UC San Diego)</br>

<img src="assets/0.gif" width="12.5%"><img src="assets/1.gif" width="12.5%"><img src="assets/2.gif" width="12.5%"><img src="assets/3.gif" width="12.5%"><img src="assets/4.gif" width="12.5%"><img src="assets/5.gif" width="12.5%"><img src="assets/6.gif" width="12.5%"><img src="assets/7.gif" width="12.5%"></br>

[[Website]](https://www.tdmpc2.com) [[Paper]](https://arxiv.org/abs/2310.16828) [[Models]](https://www.tdmpc2.com/models)  [[Dataset]](https://www.tdmpc2.com/dataset)

----


## Getting started

You will need a machine with a GPU and at least 12 GB of RAM for single-task online RL with TD-MPC**2**, and 128 GB of RAM for multi-task offline RL on our provided 80-task dataset. A GPU with at least 8 GB of memory is recommended for single-task online RL and for evaluation of the provided multi-task models (up to 317M parameters). Training of the 317M parameter model requires a GPU with at least 24 GB of memory.

We provide a `Dockerfile` for easy installation. You can build the docker image by running

```
cd docker && docker build . -t <user>/tdmpc2:1.0.1
```

This docker image contains all dependencies needed for running DMControl. We also provide a pre-built docker image [here](https://hub.docker.com/repository/docker/nicklashansen/tdmpc2/tags/1.0.1/sha256-b07d4e04d4b28ffd9a63ac18ec1541950e874bb51d276c7d09b36135f170dd93).

If you prefer to use `conda` rather than docker, start by running the following command:

```
conda env create -f docker/environment.yaml
```

The `docker/environment.yaml` file installs dependencies required for training on DMControl tasks. Other domains can be installed by following the instructions in `docker/environment.yaml`.

If you want to run ManiSkill2, you will additionally need to download and link the necessary assets by running

```
python -m mani_skill2.utils.download_asset all
```

which downloads assets to `./data`. You may move these assets to any location. Then, add the following line to your `~/.bashrc`:

```
export MS2_ASSET_DIR=<path>/<to>/<data>
```

and restart your terminal. Note that Meta-World requires MuJoCo 2.1.0 and `gym==0.21.0` which is becoming increasingly difficult to install. We host the unrestricted MuJoCo 2.1.0 license (courtesy of Google DeepMind) at [https://www.tdmpc2.com/files/mjkey.txt](https://www.tdmpc2.com/files/mjkey.txt). You can download the license by running

```
wget https://www.tdmpc2.com/files/mjkey.txt -O ~/.mujoco/mjkey.txt
```

Depending on your existing system packages, you may need to install other dependencies. See `docker/Dockerfile` for a list of recommended system packages.

----

## Supported tasks

This codebase provides support for all **104** continuous control tasks from **DMControl**, **Meta-World**, **ManiSkill2**, and **MyoSuite** used in our paper. Specifically, it supports 39 tasks from DMControl (including 11 custom tasks), 50 tasks from Meta-World, 5 tasks from ManiSkill2, and 10 tasks from MyoSuite, and covers all tasks used in the paper. See below table for expected name formatting for each task domain:

| domain | task
| --- | --- |
| dmcontrol | dog-run
| dmcontrol | cheetah-run-backwards
| metaworld | mw-assembly
| metaworld | mw-pick-place-wall
| maniskill | pick-cube
| maniskill | pick-ycb
| myosuite  | myo-key-turn
| myosuite  | myo-key-turn-hard

which can be run by specifying the `task` argument for `evaluation.py`. Multi-task training and evaluation is specified by setting `task=mt80` or `task=mt30` for the 80-task and 30-task sets, respectively. While you generally do not need to access the underlying task IDs or embeddings during training or evaluation of our multi-task models, the mapping from task name to task embedding used in our work can be found [here](https://github.com/nicklashansen/tdmpc2/blob/7ec6bc83a82a5188ca3faddc59aea83f430ab570/tdmpc2/common/__init__.py#L26). As of April 2025, our codebase also provides basic support for other MuJoCo/Box2d Gymnasium tasks; refer to the `envs` directory for a list of tasks. It should be relatively straightforward to add support for custom tasks by following the examples in `envs`.

**Note:** we also provide support for image observations in the DMControl tasks. Use argument `obs=rgb` if you wish to train visual policies.

## Repository Layout (new pieces)

Key additions on top of the original `tdmpc2` code:

- `tdmpc2/agent.py`
  - Speculative execution logic
  - Plan buffers, mismatch computation, fallback to TD-MPC2
  - Integration with corrector

- `tdmpc2/corrector.py`
  - `BaseCorrector`
  - `TwoTowerCorrector`
  - `TemporalTransformerCorrector`

- `scripts/collect_corrector_data.py`
  - Run TD-MPC2 as teacher
  - Collect distillation tuples for the corrector

- `scripts/train_corrector.py`
  - Train corrector offline on the collected dataset

- `scripts/eval_corrector.py`
  - Evaluate baseline, naive 3-step, 3-step+corrector, 6-step+corrector
  - Compare corrector architectures
