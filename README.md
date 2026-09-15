# Navid 4 NFE：8× H200 一键部署 Ref2VA

常驻 HTTP 服务，基于 **Sol-H3 + LightX2V Ref2VA 四步 LoRA**。一个任务使用全部 8 张 H200，后续任务排队。输出为 **1344×768、24 FPS、带音频的 MP4**，时长档位为 5 / 10 / 15 秒。

## 启动：只执行一个脚本

在目标 Linux 机器拉取本仓库后，进入仓库目录执行：

```bash
./deploy.sh
```

脚本自动完成：

1. 检查 NVIDIA 驱动，必要时通过 apt 安装 GCC、Git、curl 等主机工具。
2. 在本目录安装校验过 SHA256 的 uv、Python 3.12 和隔离的 `.venv`。
3. 安装 PyTorch 2.10 / CUDA 12.8、Triton、固定提交的 Diffusers 等依赖。
4. 下载固定版本的 `MiniMaxAI/MiniMax-H3` Ref2VA 分区及四步 LoRA；中断后再次执行会续传。
5. 重新检查八卡空闲显存，后台启动独立 HTTP API 和一个 `torchrun` 八卡 worker。
6. 执行真实的 5 秒 Ref2VA 生成，检查**每张卡恰好执行 4 次 DiT 前向**、MP4 视频和音频均可解码，成功后打印 `READY` 并返回。

首次执行需要联网下载依赖和大模型，模型按卡依次加载以降低主机内存峰值。可以在另一个终端用 `./deploy.sh logs` 查看进度。启动期间按 Ctrl-C 会取消本次启动；看到 `READY` 后退出终端，服务继续在后台运行。

已运行时再次执行 `./deploy.sh` 只显示现有服务状态。拉取新代码后用 `./deploy.sh restart`；若修改了 `requirements.txt`，先 `./deploy.sh stop` 再 `./deploy.sh`。

### 目标机器要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Linux x86_64；建议 Ubuntu 22.04 / 24.04，glibc ≥ 2.28 |
| GPU | 8× NVIDIA H200，SM90，建议 NVLink/NVSwitch 互联；每卡至少 125 GiB 空闲显存 |
| 驱动 | NVIDIA ≥ 570.26；脚本不会安装或替换驱动 |
| 主机内存 | 启动时至少 160 GiB 可用，建议 256 GiB 以上 |
| 磁盘 | 建议至少 250 GB 可用，用于权重、环境、缓存与生成结果 |
| 网络 | 首次安装可访问 GitHub、PyPI、PyTorch wheel 源和 Hugging Face |
| 权限 | 目录可写；缺少主机编译工具时需要 root / sudo；不需要 Docker 或 systemd |

驱动的 CUDA 12.x 小版本兼容不能覆盖所有运行时编译场景，因此本部署固定使用 CUDA 12.8 的 `ptxas`。已有的 CUDA toolkit 不需要替换。

如 Hugging Face 返回授权错误，在执行前提供拥有模型访问权限的 `HF_TOKEN`。Token 不写入代码和日志。

### 配置与已有权重复用

可选：复制 `.env.example` 为 `.env`，再调整端口、缓存路径等。环境变量优先于 `.env`。

```bash
PORT=8000 CHECKPOINT_DIR=/data/models/navid4nfe DATA_DIR=/data/navid4nfe ./deploy.sh
```

如使用环境变量配置，后续命令也需使用同一组变量；长期配置建议放 `.env`。

已有权重时设置 `MODEL_DIR` 和 `ADAPTER_PATH`，脚本校验并直接复用：

```bash
MODEL_DIR=/data/models/MiniMax-H3 \
ADAPTER_PATH=/data/models/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
./deploy.sh
```

基模目录必须是 Diffusers 布局，包含 `transformer_ref/`、`text_encoder/`、`vae/`、`audio_vae/`、`processor/`、`tokenizer/`、两个 scheduler 目录和根目录 model index。原始 `Ref2VA/transformer` 布局及只有 FL2VA 权重的目录不能直接使用。

## 调用服务

默认监听 `0.0.0.0:8000`。接口文档：`http://<机器IP>:8000/docs`。
鉴权 key 首次自动生成并保存在 `.runtime/api.key`，权限为 `0600`，重启不会改变。

```bash
export API_KEY="$(cat .runtime/api.key)"
curl --fail http://127.0.0.1:8000/readyz
```

### 1. 上传参考素材

上传**原始文件内容**，不是 multipart；每次上传一个文件，返回素材 ID：

```bash
curl --fail -X POST 'http://127.0.0.1:8000/v1/references?kind=image' \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @subject.png
```

`kind` 支持 `image`、`video`、`audio`。音频参考必须搭配图片或视频。默认每文件最大 64 MiB；图片最长边 ≤8192，视频最长边 ≤2048，宽高比均为 1:4～4:1，参考音视频长度 ≤15 秒。图片由引擎按 `match` 规则缩放。

### 2. 提交四步 Ref2VA 任务

把上传返回的 ID 按希望模型读取的顺序放入 `references`：

```bash
curl --fail -X POST http://127.0.0.1:8000/v1/videos \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A continuous cinematic shot featuring the subject in Picture 1. Natural motion and ambient sound.",
    "references": ["替换为上传返回的素材ID"],
    "duration": 5,
    "seed": 42
  }'
```

返回 HTTP 202 和任务 `id`。最多 9 张图片、3 段视频、3 段音频，总计最多 12 个参考素材。固定四步，无请求级 steps / LoRA / attention 切换。`duration` 只接受 5 / 10 / 15，省略 `seed` 时生成随机 seed，查询结果会返回实际 seed。

### 3. 查询及下载

```bash
curl --fail -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/v1/videos/任务ID

curl --fail -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/v1/videos/任务ID/content -o result.mp4
```

任务状态：`queued → running → succeeded / failed`。成功后才允许下载，查询返回 `inference_s`（不含 MP4 编码），以及包含整个任务起止时间的 `started` / `finished`。

默认最多容纳 32 个未完成任务，满队列返回 429；未就绪返回 503；无效素材或参数返回 422。CUDA/NCCL 故障或任务超过 `TASK_TIMEOUT` 会关闭 API 和全部 GPU 进程，并将未完成任务标记为失败。通过日志定位问题后执行 `./deploy.sh start`。

### 完整 HTTP 验收脚本

```bash
.venv/bin/python scripts/smoke_test.py \
  --reference image:subject.png \
  --reference audio:voice.wav \
  --duration 5 --seed 42 --output data/smoke.mp4
```

只用图片时省略音频参数。该脚本依次上传、提交、等待并下载结果；从其他机器调用时传入 `--url http://<机器IP>:8000` 和 `API_KEY` 环境变量。可再分别验证 `--duration 10`、`--duration 15`。

## 日常操作

### 启动时提示 `GPU ... has only ... GiB free`

这是模型加载前的检查，依赖安装已完成。当前 BF16/Ulysses 实现会在每张 GPU 上保留完整权重，八卡分摊序列计算，不会将权重显存除以八；不能仅降低阈值来解决显存占用。

预检查会打印全部 GPU 的总显存、已用/可用显存和 compute 进程。也可直接执行 `nvidia-smi`，确认哪些是准备替换的旧服务后，通过其原有启动器停止。若当前容器看不到占用进程，需要在宿主机查看。脚本不会自动杀死其他服务。

释放显存后再次执行 `./deploy.sh`，已安装的依赖会复用。

下载前检查通过并不表示后续显存仍然空闲。每次 `deploy/start/restart` 真正启动服务前会重新检查八卡，每个 rank 完成 CPU 权重加载后、搬入 GPU 前再检查该卡。如果 OOM 同时列出另一进程的显存占用，可用 `nvidia-smi` 和 `ps -p <PID> -o pid,ppid,user,etime,args` 确认归属；单次检查不能阻止其他服务随后抢占显存，需要确保运行期间这八张卡可持续供本服务使用。

如果 PID 持续变化或 `ps` 查不到旧 PID，执行 `./deploy.sh gpu-status`。它实时采集 GPU 进程及父进程链、工作目录、cgroup/systemd 单元线索和正在运行的 Docker 容器（含 Compose 服务名），不初始化 CUDA、不加载模型。显存检查失败时也会立即打印当时的父进程链。`ps` 查不到可能是进程已退出、权限不足或 PID 命名空间不同，不能据此判断显存已释放；应定位上层服务、watchdog 或容器，避免只停 worker 后又被拉起。

### 服务管理

```bash
./deploy.sh status     # 只有模型就绪才返回退出码 0
./deploy.sh logs       # 跟随 supervisor / API / GPU 日志
./deploy.sh errors     # 显示原始异常，无需再次下载权重或加载模型
./deploy.sh gpu-status # 实时查看 GPU 占用者、父进程链和 Docker 服务
./deploy.sh stop       # 关闭全部进程并释放 GPU
./deploy.sh start      # 复用环境和权重，启动并验证
./deploy.sh restart    # 应用当前代码和 .env
./deploy.sh check      # 停止服务后检查 CUDA、Triton 和权重
```

生成文件与素材保存在 `DATA_DIR`（默认 `data/`），任务元数据保存在同目录的 SQLite 数据库。服务不会自动删除历史文件；请按使用量安排磁盘清理。清空整个 `DATA_DIR` 前先停止服务，避免删除正在使用的素材。缓存、环境、API key、权重和数据都已加入 `.gitignore`。

这是宿主机上的后台进程，不安装系统开机服务。机器重启后执行 `./deploy.sh start`。

如日志末尾只有 `ChildFailedError`、`exitcode: 1` 和其他 rank 的 `SIGTERM`，执行 `./deploy.sh errors` 查看前面的原始 Python 异常。新启动会按 run/rank 单独保存 traceback，自动失败输出优先展示最早记录的异常；旧日志也可提取 traceback 上下文。SIGTERM 通常是某个 rank 失败后其余进程的清理结果，不能单凭它判断根因。

## H200 配置与验证边界

- 使用专用 `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`，不是 FastH3 的 T2VA adapter。
- **五个 scheduler 点对应四次 DiT 前向**。保留上游 Ref2VA 的 video/audio shift `12 / 3`、LoRA alpha `8`、AdaLN 预计算、融合算子和八卡 Ulysses。
- 默认 **Dense attention + BF16 计算/通信**。官方 B300 的 SOL/BSA、INT8 通信和 MXFP8 配置不直接作为 H200 默认配置，因此不能套用官网的 B300 耗时。
- VAE 使用按 clip 的八卡 tile 并行，默认关闭 `torch.compile`，可通过 `VAE_COMPILE=1` 开启并重新验收。
- 上游仅记录 B300 的硬件验证。本仓库在无 GPU 的开发环境验证接口、任务队列、进程失败清理及依赖解析；**尚未在真实 8× H200 上完成端到端验证，也没有 H200 性能或画质数据**。
- 自动启动检查覆盖 5 秒、单张图片参考以及视频/音频输出；多参考、视频输入、音频输入和 10/15 秒需在目标机器进一步验收。首个较长任务仍可能有额外开销，复杂参考组合可能占用更多显存。

## 固定来源

| 组件 | 固定版本 |
| --- | --- |
| Sol-H3 | NVlabs/Sana `8e0db4fa562d727ea28b8d63015c196db7d97cae`，源码随仓库提供 |
| Diffusers | `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` |
| MiniMax-H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| LightX2V Turbo | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` |
| PyTorch / Triton | `2.10.0+cu128` / `3.6.0` |

来源：[Sol-H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3)、[LightX2V LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo)、[PyTorch CUDA 12.8 安装说明](https://pytorch.org/get-started/previous-versions/#v2100)。上游原始文件哈希和本地改动说明位于 `vendor/sol_h3/`。

本地 CPU 测试：安装 `fastapi`、`httpx`、`Pillow`、`numpy`、`av` 和 `pytest` 后执行 `python -m pytest -q tests`。`requirements.lock` 锁定了 Linux x86_64 / Python 3.12 的 74 个运行时依赖；修改版本时应同步重新解析该文件。
