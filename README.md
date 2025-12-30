# gym-v

## 开发前可以先安装一下pre-commit (uv run pre-commit install)

## 阶段一

我们先给一些已经有的env集成进来，新增的env需要：
- 和原仓库的env的游戏行为一致
- 代码也尽量和原仓库保持一致（方便review）
- 如果原仓库是纯文本env，集成进来需要新增出图函数并且可以注意一下game description部分不要再包含原来的与纯文本caption / maze相关的内容
- 如果是纯文本env，在实现时候以注释形式写一下原本文本env的一个qa的case，新的game description还需要注意需要与文本env预期的回答format和现在的render的图对的上（可参考下rg在gym-v得写法

## Offline data 接入（单轮）

gym-v 提供了一个通用的 offline 单轮环境：**`Offline/SingleTurn-v0`**。它从 JSONL 数据集中读取样本，在 `reset()` 时产出 `Observation(image, text, metadata)`，在 `step(action)` 时用 grader 对 action 判分并终止。

### 用法

- 先确保注册内置 env：
  - `import gym_v.envs`
- 然后创建 env：
  - `gym_v.make("Offline/SingleTurn-v0", dataset_path=".../dataset.jsonl")`

### JSONL schema

每行一个 JSON object（**必须是 object**，不是 array / string）。字段规范如下（single-turn）：

- **必需条件**：`text` 和 `image_path` **至少存在一个**（都缺失会报错）。
- **text**: `string | null`
  - 不带图任务通常只用 `text` 即可。
- **image_path**: `string | null`
  - 支持绝对路径；相对路径会相对该 `.jsonl` 所在目录解析成绝对路径。
  - 默认会校验文件存在（如需跳过，可在创建 env 时传 `validate_files=False`）。
- **answer**: `string | null`
  - 用于 `step(action)` 的判题；没有 `answer` 时 env 会返回 `reward=0`，并在 info 里标注原因。
- **metadata**: `object | null`
  - 任意附加信息（如 `id`、`difficulty`、`source`、原始字段等）。

默认 grader：`exact_match`（大小写/空白归一化后精确匹配，正确 reward=1.0，否则 0.0）。

### sampling 说明

- `sampling="sequential"`：顺序遍历
- `sampling="shuffle"`：每个 epoch 生成一份随机排列

### 最小样例

**1) 不带 image（纯文本）**

```json
{"text":"Q: 2+2=?","answer":"4","metadata":{"id":"q1"}}
{"text":"Q: 2+2=?","answer":"4","metadata":{"id":"q1"}}
```

**2) 带 image（可选 text）**

```json
{"image_path":"images/000001.png","text":"Describe the image.","answer":"a red square","metadata":{"id":1}}
```
