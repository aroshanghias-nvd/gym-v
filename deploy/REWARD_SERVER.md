GenEval Reward Server 说明

目标
- 把 GenEval 评测从本地逻辑中拆出来，独立部署成服务。
- 通过 Ray Serve 启动（`gym_v.deploy.rewards.deploy`）。
- 与 `gym_v.envs.eval.t2ieval.geneval_env` 的 OpenAI 兼容 JSON 请求对齐。

文件结构
- `gym_v/deploy/rewards/server.py`：Ray Serve 服务实现（OpenAI 兼容 JSON）。
- `gym_v/deploy/rewards/deploy.py`：部署入口。
- `gym_v/deploy/rewards/local/geneval/`：GenEval 本地 reward 实现与模型依赖。
- `start.sh`：最简启动脚本（默认 geneval）。

接口路径
- `POST /v1/chat/completions`：OpenAI 兼容 JSON 协议（含 image_url data URL）。

启动方式
```
./start.sh
```

环境变量
- `SCORE_JSON`：reward 配置，默认 `{"geneval":{...}}`（见 `start.sh`）。
- `DEVICE`：cuda/cpu（默认 cuda）。
- `PORT`：默认 18085。
- `NUM_GPUS`：每个实例占用的 GPU 数量（默认 1，可设为 >1）。
- `NUM_REPLICAS`：可选，指定实例数；不设置则根据 Ray 集群 GPU 总数和 `NUM_GPUS` 自动计算。
- `RANK`/`NODE_RANK`：多节点时用于区分 head/worker，rank=0 为 head（torchrun 会自动设置 `RANK`）。
- `MASTER_ADDR`/`RAY_HEAD_ADDR`：worker 连接 head 的地址（torchrun 会自动设置 `MASTER_ADDR`）。
- `RAY_PORT`：Ray 端口，默认 6379。
- `WORKER_HOLD`：worker 启动后是否阻塞等待（默认 0，设为 1 则 `tail -f /dev/null`）。
- `MAX_ONGOING_REQUESTS`：可选，Ray Serve 的 `max_ongoing_requests`（start.sh 透传）。

启动参数（deploy.py）
- `--max-ongoing-requests`：Ray Serve 单副本最大并发请求数（默认 5120）。

示例（传入 Geneval 模型路径）
```
SCORE_JSON='{"geneval":{"init":{"config_path":"...","ckpt_root":"...","object_names_path":"..."}}}' ./start.sh
```

请求格式（OpenAI 兼容 JSON，单图示例）
```
curl -s http://127.0.0.1:18085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "geneval",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
          {"type": "text", "text": "score this image"}
        ]
      }
    ],
    "metadata": {
      "meta_datas": [ ... ],
      "only_strict": true
    }
  }'
```

说明
- `messages` 支持 OpenAI 兼容格式：`image_url` + `text`。
- `metadata` 必须是 dict，会直接传给 reward 的 `score(..., metadata=...)`。
- `metadata` 内若包含 base64 或图片路径，会尝试解码为 PIL.Image。
- 不需要显式传 `input_format`，服务端会自动识别。

批处理合并逻辑（Ray Serve batch）
- 多个请求会被合并成一个大 batch 调用 `scorer`。
- 图片与 prompt 会顺序拼接后一次性送入 reward。
- `metadata` 按 key 合并：
  - 如果 value 是 list/tuple/dict：按请求 append，得到 list-of-requests（例如 `meta_datas` -> `[[...],[...]]`）。
  - 其它标量 key：使用第一条请求的值。

示例（两个请求合并）
```
req1.metadata = {"meta_datas": [a, b], "only_strict": true, "cfg": {"foo": 1}}
req2.metadata = {"meta_datas": [c], "only_strict": true, "cfg": {"foo": 1}}
merged.metadata = {
  "meta_datas": [[a, b], [c]],
  "only_strict": true,
  "cfg": [{"foo": 1}, {"foo": 1}]
}
```
