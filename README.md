# Navid 4/8 NFE：8× H200 一键部署 Ref2VA

常驻 HTTP 服务，基于 **Sol-H3 + LightX2V Ref2VA Turbo**，支持四步和八步，**默认八步 v1.0 768p LoRA**。一个任务使用全部 8 张 H200，后续任务排队。输出为 **24 FPS、带音频的 MP4**，默认 1344×768；支持 **4～15 整数秒、自定义横竖版分辨率、参考图短边、请求级 Sol attention / DiT 缓存**。默认启用 **Sol τ=1.5、DiT cache 和 DiT torch.compile**，VAE 编译关闭；Sol / DiT 序列容量固定按 4096 tokens 对齐。

## 启动：只执行一个脚本

在目标 Linux 机器拉取本仓库后，进入仓库目录执行：

```bash
./deploy.sh
```

脚本自动完成：

1. 检查 NVIDIA 驱动，必要时通过 apt 安装 GCC、Git、curl 等主机工具。
2. 在本目录安装校验过 SHA256 的 uv、Python 3.12 和隔离的 `.venv`。
3. 安装 PyTorch 2.10 / CUDA 12.8、Triton、固定提交的 Diffusers 等依赖。
4. 下载固定版本的 `MiniMaxAI/MiniMax-H3` Ref2VA 分区及所选档位的 LoRA，并校验 LoRA 的 SHA256；中断后再次执行会续传。
5. 重新检查八卡空闲显存，后台启动独立 HTTP API 和一个 `torchrun` 八卡 worker。
6. 在每张 H200 上执行 Sol/TMA 数值检查（包含 4096 桶边界、部分尾块），再按服务默认优化配置执行真实的 5 秒 Ref2VA 生成，检查**每张卡恰好执行所选的 4 或 8 次 DiT 前向**、MP4 视频和音频均可解码，成功后打印 `READY` 并返回。

首次执行需要联网下载依赖和大模型。模型加载默认按可用主机/容器内存自动选择 1 / 2 / 4 / 8 个 rank 并发；约 2 TB 空闲内存的主机可八卡同时加载。可以在另一个终端用 `./deploy.sh logs` 查看进度。启动期间按 Ctrl-C 会取消本次启动；看到 `READY` 后退出终端，服务继续在后台运行。

等待提示和 `./deploy.sh status` 会显示本次启动最近的加载进度，GPU rank 从 0 到 7。`MODEL_LOAD_PLAN` 显示并发数和内存预算，`Loading model` 表示开始该卡加载，`CPU weights ready` 表示 CPU 权重准备完成，`Loaded model` 显示 CPU 加载和 CUDA 搬运各自耗时；八卡加载后还有 LoRA 融合和真实生成预热。仅重复出现 `loading` 不能说明卡死，需结合 worker 日志和 GPU 占用判断。`READY_TIMEOUT` 默认 7200 秒，是启动超时上限，不是预计耗时。

`restart` 会退出旧进程、释放显存，因此每次都要重新把完整模型装到每张卡，并重新融合 LoRA / 建立 AdaLN 表 / 执行预热。磁盘编译缓存能复用内核，不能保留已退出进程的 GPU 权重。模型常驻后，连续生成任务不重复加载；参考大小、时长、分辨率、Sol/cache 请求参数均不需要重启。

`MODEL_LOAD_PARALLELISM=auto`（默认）按当前可用 RAM 选择并发数：每个加载中的 rank 预留 192 GiB、额外留 64 GiB；并发 8 需要至少 1600 GiB 可用。标准 cgroup v1/v2 限制也纳入判断。可显式指定 1 / 2 / 4 / 8；1 恢复串行，超出预算拒绝启动。并发加载共享磁盘和内存带宽，实际提速以日志为准。每卡仍保留完整权重，GPU 显存要求不变。

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

### 四步 / 八步档位

| `.env` 配置 | 对应 LoRA | 采样配置 |
| --- | --- | --- |
| `REF2VA_NFE=8`（默认） | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | Euler，video/audio shift 12/3，9 个 scheduler 点、8 次 DiT 前向 |
| `REF2VA_NFE=4` | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | Euler，video/audio shift 12/3，5 个 scheduler 点、4 次 DiT 前向 |

修改 `.env` 的 `REF2VA_NFE` 后执行 `./deploy.sh restart`。`start/restart` 会自动下载缺失的所选 LoRA（约 1.38 GB），复用已有基模；已有正确 LoRA 则直接校验并复用。两档使用不同权重和 AdaLN 表，因此切换需要重启。请求可传 `nfe:8` 或 `nfe:4` 检查当前档位，不能在请求中切换权重；与当前档位不同会返回 422。

升级后未设置 `REF2VA_NFE` 的部署会使用八步。如果旧 `.env` 显式设置了四步 `ADAPTER_PATH`，需删除该项让脚本自动选择，或改成八步文件；文件哈希与档位不符会在加载 GPU 权重前报错，保留原文件。`/readyz` 的 `nfe`、`profile` 以及 `.runtime/checkpoints.json` 记录实际档位和权重身份；API 与 worker 档位不一致时不会报告 ready。

八步配置来自 [LightX2V 作者的 Ref2V 8-step v1.0 发布说明](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51)。该版本发布范围为 768p；接口仍允许其他分辨率，效果需另行验证。八步是否改善人群音效，需要同素材、提示词、seed、尺寸下与四步对照试听，不能仅凭步数保证。

默认启用 Sol、DiT cache 和 DiT 编译。升级已有部署时，将 `.env` 中相关项设为（只需修改一次，后续直接 start/restart）：

```bash
REF2VA_NFE=8
SOL_ATTN_ENABLED=1
SOL_ATTN_TAU=1.5
SOL_ATTN_DENSE_STEPS=1
CACHE_DIT_ENABLED=1
DIT_COMPILE=1
VAE_COMPILE=0
```

然后执行 `./deploy.sh restart`。一次性覆盖旧环境的等价命令：

```bash
REF2VA_NFE=8 SOL_ATTN_ENABLED=1 SOL_ATTN_TAU=1.5 SOL_ATTN_DENSE_STEPS=1 \
CACHE_DIT_ENABLED=1 DIT_COMPILE=1 VAE_COMPILE=0 ./deploy.sh restart
```

已有权重时设置 `MODEL_DIR` 和 `ADAPTER_PATH`，脚本校验并直接复用：

```bash
MODEL_DIR=/data/models/MiniMax-H3 \
ADAPTER_PATH=/data/models/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors \
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

### 2. 提交 Ref2VA 任务

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

返回 HTTP 202 和任务 `id`。最多 9 张图片、3 段视频、3 段音频，总计最多 12 个参考素材。省略 `nfe` 使用服务当前档位（默认八步）；显式 `nfe` 仅校验档位。`duration` 接受 4～15 整数，省略 `seed` 时生成随机 seed，查询结果会返回实际 seed。

### 可选调优参数

```json
{
  "prompt": "The man in Picture 1 walks out of a hotel and waves to the crowd. Cheering and city ambience.",
  "references": ["参考图ID"],
  "duration": 8,
  "resolution": 768,
  "ratio": "9:16",
  "seed": 42,
  "reference_short_edge": 512,
  "optimization": {
    "sol_attn": {
      "enabled": true,
      "tau": 1.5,
      "dense_steps": 1,
      "sink_conditioning": "exact_kv_and_rows",
      "dense_prefix_seconds": 0
    },
    "cache_dit": {
      "enabled": true,
      "warmup": 1,
      "rdt": 0.08,
      "max_continuous_cached_steps": 1
    }
  }
}
```

| 参数 | 取值和行为 |
| --- | --- |
| `reference_short_edge` | 128～2048 整数，等比例缩放后各边取最近的 32 倍数；默认只缩小，`reference_allow_upscale:true` 允许放大小图。省略则保留原来的约 1MP 上限。实际处理尺寸回传；无扩图。 |
| `duration` | 4～15 整数。按 `17*n+5` 原生帧数推理，再截取前 `24*duration` 帧及对应音频，不变速；8 秒恰好原生 192 帧。AAC 容器可能有不足一帧编码填充。 |
| `width` / `height` | 成对传入，128～4096 偶数，宽高比 1:4～4:1。内部各边向上取 32 倍数，输出缩放为请求尺寸（不会裁掉画面）。 |
| `resolution` / `ratio` | 可替代 width/height；resolution 为短边（128～2048 偶数），ratio 支持 16:9、9:16、1:1、4:3、3:4、21:9、9:21。两组参数不能混用。 |
| `sol_attn.enabled` | `false` 为 Dense，`true` 使用 H200 的 `triton_tma_sm90` Sol 稀疏注意力。不会静默切换 CuTe/BSA 后端。 |
| `sol_attn.tau` | `(0,10]`，越大通常越稀疏，质量损失可能越大；当前默认 1.5。 |
| `sol_attn.dense_steps` | 0～当前 NFE，前多少个采样步使用 Dense；等于当前 NFE 时全程 Dense。 |
| `sol_attn.sink_conditioning` | `exact_kv`：保留全部 text/reference/audio 前缀 KV；`exact_kv_and_rows`：同时精确计算前缀 query；`off`：不保护前缀。KV 保护边界向外对齐 64。 |
| `sol_attn.dense_prefix_seconds` | 0～15，目标视频开头对应的 latent 帧使用 Dense query，边界向上取整到 latent 帧；覆盖整段时全程 Dense。 |
| `cache_dit.enabled` | 开关跨步残差缓存；本地实现 Cache-DiT 的 DBCache/Fn=1、Bn=0 策略，非额外安装官方 cache-dit 包。 |
| `cache_dit.warmup` | 1～当前 NFE，前多少个采样步全算；最后一个采样步始终全算。 |
| `cache_dit.rdt` | `[0,1]`，累计相对 L1 变化阈值，越高越容易复用；0 不命中。0.08 为保守起始值，四步模型可能一次都不命中。 |
| `cache_dit.max_continuous_cached_steps` | 四步档 1～2，八步档 1～6，默认均为 1；允许连续复用的中间步数，每步仍计算第一个探测 block。 |

省略的调优字段继承 `.env` 默认值，显式 `enabled:false` 可以关闭。每个任务重新初始化缓存与配置，八卡通过全局归约决定是否复用，前一任务不会污染下一任务。参考算法：[Cache-DiT DBCache](https://github.com/vipshop/cache-dit/blob/main/docs/user_guide/DBCACHE_DESIGN.md)。

默认开启 Sol（τ=1.5、首步 Dense、`exact_kv_and_rows`）和 DiT 缓存（warmup=1、rdt=0.08、连续复用上限=1）；启动预热与省略参数的请求使用同一配置。显式请求可分别关闭 Sol 或缓存用于对照。缓存开启不代表每个任务都会命中；两档的首步和末步始终全算。开启优化并不保证加速；需要观察实际 sparse calls、缓存命中和画质。H200 当前选择已有的 Sol Triton/TMA 路径并增加 runtime length 支持，没有直接套用仅支持 SM100/103 的 CuTe 编译桶。

### 编译与资源上限

- `DIT_COMPILE=1` **默认开启**，复用 PyTorch 编译图和 Inductor 磁盘缓存；`VAE_COMPILE=0` 默认关闭。显式设置 `DIT_COMPILE=0` 并重启即可使用 eager DiT，不影响请求级 Sol / DiT cache 开关。
- packed sequence 始终向上对齐 **4096**，八卡分片均匀；额外行在注意力入口排除，不能作为 KV。提示词文本、seed、参考素材 ID 均不进入编译键。
- Sol 的描述符容量和 autotune 按 4096 分桶，真实长度及 tau 是运行时参数。启用 `DIT_COMPILE=1` 时编译 DiT 数值部分，通信和注意力调度仍保留 eager。显式打开 VAE 编译时按实际 tile 形状编译，不能仅凭总 token 数复用。
- 旧 `.env` 中的 `DIT_COMPILE=0`、`CACHE_DIT_ENABLED=0` 等显式值仍会覆盖新默认值。升级时将相关项改成上文配置；不要直接用示例覆盖整个 `.env`，以免丢失路径、端口或凭据。
- Sol 使用 `.runtime/triton-cache`，DiT 编译使用 `.runtime/inductor-cache`，两者均跨重启保留。首次遇到新桶仍可能有编译/调优开销；已有磁盘缓存也不保证跳过新进程的全部图捕获和预热。
- `/readyz` 的 `capabilities.compilation` 显示 DiT/VAE 编译开关；`capabilities.optimization_defaults` 显示 Sol 和 DiT cache 默认值。启动预热的实际指标保存到 `.runtime/warmup-metrics.json`。
- `metrics.compile` 分别记录 `shape_seen`（过去成功执行过该形状）、`inductor_invocations`、`inductor_compile_s`、`graph_reused`；不把“见过形状”冒充底层磁盘编译缓存命中。PyTorch 调用编译后端的耗时可能包含磁盘缓存加载。
- `MAX_OUTPUT_PIXELS=2088960`（可容纳 1920×1088），`MAX_PACKED_TOKENS=262144`。先按输出/参考素材组合估算，再按真实 packed rows 检查。它们是准入限制，不是所有组合都能放进显存的保证；超限先降低分辨率、时长或参考大小。

### 3. 查询及下载

```bash
curl --fail -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/v1/videos/任务ID

curl --fail -H "X-API-Key: $API_KEY" \
  http://127.0.0.1:8000/v1/videos/任务ID/content -o result.mp4
```

任务状态：`queued → running → succeeded / failed`。成功后才允许下载，查询返回 `execution`（解析后的配置和权重档位）、`metrics`（参考图尺寸、Sol 实际后端/调用数/首个调用 head0 的路由密度、DiT 缓存命中、编译及阶段耗时）。`nfe` 指四个或八个采样前向，缓存会减少内部 block 计算；历史任务保留原有 NFE，不随服务切档改变。`inference_s` 是 GPU pipeline 时间，不含输出尺寸调整、MP4 编码、上传下载；`started/finished` 包含整个任务执行。GPU 阶段计时含首次编译等待，conditioning 包含参考图和文本编码。

默认最多容纳 32 个未完成任务，满队列返回 429；未就绪返回 503；无效素材或参数返回 422。CUDA/NCCL 故障或任务超过 `TASK_TIMEOUT` 会关闭 API 和全部 GPU 进程，并将未完成任务标记为失败。通过日志定位问题后执行 `./deploy.sh start`。

### 完整 HTTP 验收脚本

```bash
.venv/bin/python scripts/smoke_test.py \
  --reference image:subject.png \
  --reference audio:voice.wav \
  --duration 5 --seed 42 --output data/smoke.mp4
```

只用图片时省略音频参数。该脚本依次上传、提交、等待并下载结果；从其他机器调用时传入 `--url http://<机器IP>:8000` 和 `API_KEY` 环境变量。省略优化开关时继承服务默认值；`--no-sol`、`--no-cache-dit` 可显式关闭。可添加 `--duration 8 --width 768 --height 1366 --reference-short-edge 512 --sol --tau 1.5 --no-cache-dit`，脚本会保存 MP4 和同名 JSON 指标。

模型服务就绪后执行 `./deploy.sh verify`：使用同一素材/提示词/seed、1344×768 比较 Dense、Sol 1.5、Cache、两者同时开启，然后再次关闭优化；另测 4 秒竖版 1080p、15 秒正方形 512。结果按当前档位写入 `data/tuning-validation/nfe4/` 或 `nfe8/`。可用 `.venv/bin/python scripts/verify_tuning.py --reference subject.png --nfe 8` 指定真人参考并校验档位；`smoke_test.py` 也支持 `--nfe`。脚本验证媒体格式/时长、实际 NFE 和优化是否执行，身份、动作、音频质量需要观看视频评估。

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

- 使用所选档位的专用 Ref2VA 四步 v0.1 / 八步 v1.0 768p LoRA，固定文件 SHA256，禁止与 FL2VA / T2VA adapter 混用。
- **五个 / 九个 scheduler 点对应四次 / 八次 DiT 前向**。两档采用 video/audio shift `12 / 3`、LoRA alpha `8`，保留 AdaLN 预计算、融合算子和八卡 Ulysses；AdaLN 表长度必须匹配当前 NFE。
- 默认 **Sol attention（τ=1.5）+ BF16 计算/通信**，首步仍为 Dense；默认启用 DiT cache / DiT 编译，关闭 VAE 编译。官方 B300 的 SOL/BSA、INT8 通信和 MXFP8 配置不直接作为 H200 默认配置，因此不能套用官网的 B300 耗时。
- VAE 使用按 clip 的八卡 tile 并行，默认关闭 `torch.compile`，可通过 `VAE_COMPILE=1` 开启并重新验收。
- 根据 2026-09-15 目标机器返回的 `READY` 日志，**8× H200 已通过内置 5 秒、单张图片 Ref2VA 启动预热**：每卡执行四次 DiT 前向，生成的视频和音频可解码，HTTP 就绪检查通过。开发环境另验证了接口、任务队列、进程失败清理及依赖解析。
- 2026-09-15 的实机单图、5 秒 HTTP 任务已完成上传、提交、查询及下载验证。下载后的 MP4 完整解码通过：1344×768、24 FPS、124 帧、H.264 视频和 32 kHz AAC 音轨。该次预热后任务的 `inference_s` 为 5.973 秒，任务执行时间为 6.916 秒（不含客户端上传、下载）；这是单次观测，不是通用性能基准。
- 2026-09-15 另已完成 33 条 768×1366、8–15 秒的 Ref2VA 实机任务，使用 Sol τ=1.5、DiT cache 关闭、DiT 编译开启；视频和音频解码通过。该批结果的 DiT cache 未开启，不能作为本次默认缓存配置的性能或质量依据。
- 以上实机记录均为四步，八步档及其 Sol + DiT cache 默认组合尚未在目标 GPU 上验收。本地测试覆盖两档调度器、缓存步数、档位校验和历史记录；目标机启动会执行真实八卡预热，再用 `./deploy.sh verify` 做端到端验收。多参考、视频输入、音频输入仍需进一步验收；当前没有正式的 H200 画质评测数据。首个较长任务仍可能有额外开销，复杂参考组合可能占用更多显存。

## 音频质量排查

输出音频沿用 Audio VAE 的原生 32 kHz 双声道，再编码为 AAC；视频尺寸调整不会改变音频采样率，时长处理只截取对应样本、不变速。媒体解码通过仅验证格式和时长，不代表人声、音效的主观质量通过。

人群场景应明确声音层次：远处低音量人群底噪、少量短促欢呼、近处清楚的脚步/快门声，避免同时要求持续大声欢呼、多人对白、广播、引擎和交响乐。H3 提示词可分别用 `overall_soundscape` 与 `non_diegetic_music` 描述环境音和配乐，参见[官方 Ref2VA 提示词指南](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt)。修改提示词仍需试听验证，不能把嘈杂直接认定为削波或稀疏注意力问题。

定位时固定素材、提示词、seed、尺寸和时长，先显式关闭 `cache_dit.enabled`，比较 `sol_attn.enabled=false` 与 Sol 1.5，再单独改声音提示词。当前 `exact_kv_and_rows` 会对音频前缀执行 Dense query，但音视频共用 Transformer，视频侧近似仍可能间接影响声音。比较四步 / 八步时通过 `REF2VA_NFE` 切换对应 LoRA 并重启，使用完全相同的请求输入；两档均使用 video/audio shift 12/3。不要混用 FL2VA 八步权重或 FL2VA 768p 的 video shift 6。

## 固定来源

| 组件 | 固定版本 |
| --- | --- |
| Sol-H3 | NVlabs/Sana `8e0db4fa562d727ea28b8d63015c196db7d97cae`，源码随仓库提供 |
| Diffusers | `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` |
| MiniMax-H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| LightX2V Turbo | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` |
| PyTorch / Triton | `2.10.0+cu128` / `3.6.0` |

来源：[Sol-H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3)、[LightX2V LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo)、[PyTorch CUDA 12.8 安装说明](https://pytorch.org/get-started/previous-versions/#v2100)。上游原始文件哈希和本地改动说明位于 `vendor/sol_h3/`。

本地 CPU 测试：安装 `fastapi`、`httpx`、`Pillow`、`numpy`、`av`、`torch`、`safetensors`、`huggingface_hub`、上述固定版本的 Diffusers 和 `pytest` 后执行 `python -m pytest -q tests`。`requirements.lock` 锁定了 Linux x86_64 / Python 3.12 的 74 个运行时依赖；本次档位切换无需修改依赖。
