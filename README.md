# 基于 Lerobot 与 SmolVLA 的机械臂视觉语言操作 Demo

## 概述

本项目基于 Lerobot 框架与 SmolVLA 模型，实现了机械臂在视觉语言指令下的多任务抓取与放置操作。通过收集多样化的演示数据、微调预训练模型，并在真实机械臂上进行了部署与推理测试，验证了模型在有限数据下的泛化能力与任务理解能力。

---

## 目录

- [环境配置](#环境配置)
- [数据收集](#数据收集)
- [模型训练](#模型训练)
- [模型推理与测试](#模型推理与测试)
- [实验结果与分析](#实验结果与分析)
- [常见问题与解决](#常见问题与解决)
- [总结与展望](#总结与展望)
- [附录：完整命令集](#附录完整命令集)

---

## 环境配置

### 硬件
- **机械臂**：SO101（两台，分别作为 follower 和 leader）
- **相机**：2 × 640×480 分辨率，分别置于腕部（handeye）和固定俯视位置（fixed）
- **主机**：
  - 本机：Ubuntu 22.04, RTX 5060 8GB
  - 服务器：Ubuntu 22.04, A6000 48GB

### 软件
- **Lerobot 版本**：0.4.1（确保本机与服务器版本一致）
- **Python 版本**：3.10
- **CUDA**：>= 11.8

### 环境准备
```bash
# 安装 Lerobot
pip install lerobot

# 查看相机
lerobot-find-cameras

# 赋予串口权限
sudo chmod 666 /dev/ttyACM*
```

---

## 数据收集

两组任务描述

1. pick the black tape and place it on  the pink plate
2. pick the yellow banana and place it on  the pink plate

设计了7组不同场景的抓取任务，共收集90条 episode 数据，任务包括：

1. **仅香蕉固定位置固定方向** → 15条
2. **仅胶带固定位置** → 15条
3. **香蕉与胶带共存，固定位置，固定香蕉方向，抓香蕉** → 15条
4. **香蕉与胶带共存，固定位置，固定香蕉方向，抓胶带** → 15条
5. **仅香蕉，方向变化，位置固定** → 10条(有5个标记好的方向，所以是一个方向两条数据)
6. **香蕉与胶带共存，抓香蕉，香蕉方向变化** → 10条
7. **香蕉与胶带共存，抓胶带，香蕉方向变化** → 10条

![](\images\image1.jpg)

![](\images\image2.jpg)

![](\images\image3.jpg)

胶带和香蕉位置固定标记

### 数据收集命令示例（第一组）

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=R12253204 \
    --robot.cameras="{'handeye': {'type':'opencv', 'index_or_path':4, 'width':640, 'height':480, 'fps':25}, 'fixed': {'type':'opencv', 'index_or_path':6, 'width':640, 'height':480, 'fps':25}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=R07253204 \
    --display_data=true \
    --dataset.repo_id=Lisette1231/20251216so101_1 \
    --dataset.num_episodes=15 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="pick the yellow banana and place it on the pink plate" \
    --dataset.push_to_hub=true
```

### 合并数据集
```bash
lerobot-edit-dataset \
    --repo_id Lisette1231/20251217so101_merged \
    --operation.type merge \
    --operation.repo_ids "['Lisette1231/20251216so101_1', 'Lisette1231/20251216so101_2','Lisette1231/20251216so101_3','Lisette1231/20251216so101_4','Lisette1231/20251216so101_5','Lisette1231/20251216so101_6','Lisette1231/20251216so101_7']"
```

上传至 Hugging Face Hub：
```bash
hf upload Lisette1231/20251216so101_merged \
    ~/.cache/huggingface/lerobot/Lisette1231/20251216so101_merged \
    --repo-type dataset
```

---

## 模型训练

### 关键发现
- **必须加载预训练权重**：仅指定 `policy.type=smolvla` 会导致模型为空架子，推理时机械臂抖动。
- **正确命令**：应同时指定 `policy.path=lerobot/smolvla_base`。

只设置policy_type=smolvla训练出来的模型推理只会发电

### 训练命令
```bash
lerobot-train \
  --dataset.repo_id=Lisette1231/20251217so101_merged \
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs/train/20251218so101smolvla \
  --job_name=20251218so101smolvla \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=Lisette1231/20251218so101smolvla_addpath \
  --policy.push_to_hub=true \
  --steps=30000 \
  --batch_size=32 \
  --save_freq=10000
```

### 恢复训练
```bash
lerobot-train \
  --config_path=outputs/train/20251218so101smolvla/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --policy.push_to_hub=false
```

### 上传训练好的模型
```bash
hf upload Lisette1231/20251218so101smolvla_last_addpath \
  outputs/train/20251218so101smolvla/checkpoints/last/pretrained_model
```

---

## 模型推理与测试

我们进行了多组推理测试，涵盖：
- 单一物体抓取（香蕉/胶带）
- 多物体选择抓取
- 物体方向与位置变化
- 任务语言泛化测试

### 推理命令示例
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=R12253204 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=R07253204 \
  --robot.disable_torque_on_disconnect=true \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path':4, 'width': 640, 'height': 480, 'fps': 25}, 'fixed': {'type': 'opencv', 'index_or_path': 6, 'width': 640, 'height': 480, 'fps': 25}}" \
  --display_data=true \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=40 \
  --dataset.num_episodes=5 \
  --dataset.single_task="pick the yellow banana and place it on the pink plate" \
  --policy.path=Lisette1231/20251218so101smolvla_last_addpath \
  --policy.device=cuda \
  --dataset.repo_id=Lisette1231/eval_2025120so101smovla_onlybananafor5 \
  --dataset.push_to_hub=true
```

### 视频

#### 1. only香蕉

![1](https://github.com/1231Lisette/smolvla_demo/raw/main/videos/only_banana.mp4)

## 实验结果与分析

### 香蕉抓取任务成功率
| 场景                       | 测试集数 | 成功率       | 备注                                    |
| -------------------------- | -------- | ------------ | --------------------------------------- |
| 仅香蕉固定位置方向         | 5        | 5/5 (100%)   | 启动偶有延迟                            |
| 香蕉与胶带共存（固定方向） | 10       | 10/10 (100%) | 连续抓取表现稳定                        |
| 香蕉方向变化               | 10       | 9/10 (90%)   | 一处未见过方向失败                      |
| 香蕉随机位置方向           | 5        | 4/5 (80%)    | 泛化能力尚可                            |
| 语言泛化测试               | 4        | 4/4 (100%)   | 对“yellow one”、“plate”等泛化词理解良好 |

### 胶带抓取任务成功率
| 场景                       | 测试集数 | 成功率     | 备注                 |
| -------------------------- | -------- | ---------- | -------------------- |
| 仅胶带固定位置             | 5        | 3/5 (60%)  | 启动不稳定，抖动明显 |
| 香蕉与胶带共存（固定方向） | 5        | 5/5 (100%) | 启动偶需人工干预     |
| 香蕉方向变化时抓胶带       | 5        | 5/5 (100%) | 启动有时延迟         |
| 香蕉随机位置时抓胶带       | 4        | 4/4 (100%) | 表现稳定             |
| 胶带随机位置               | 5        | 1/5 (20%)  | 泛化能力差           |

### 关键观察
1. 模型对训练过位置与方向泛化良好
2. 胶带抓取启动稳定性差于香蕉
3. 视觉语言理解具有一定泛化性，可适应不同表达方式
4. 训练数据中未出现的位置与方向组合表现较差

---

## 常见问题与解决

### 1. 机械臂抖动
- **问题**：推理时机械臂抖动严重
- **原因**：训练时未加载预训练权重，仅使用 `policy.type=smolvla`
- **解决**：训练时添加 `--policy.path=lerobot/smolvla_base`

### 2. 推理时不启动
- **现象**：模型等待时间过长，需人工干预（手在相机前晃动）
- **可能原因**：视觉特征未触发动作生成
- **临时解决**：增加环境运动以触发模型响应

### 3. 跨版本兼容性问题
- **建议**：保持本机与服务器 Lerobot 版本一致，避免模型与代码不匹配

### 4.视觉有在起作用吗？

执行抓香蕉的任务，我的桌子上已经没有香蕉了，可是smolvla一直继续抓？

