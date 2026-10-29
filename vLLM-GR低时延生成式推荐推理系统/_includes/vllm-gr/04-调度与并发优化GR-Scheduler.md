# 4 调度与并发优化 GR-Scheduler

生成式推荐通常只生成几段 SID，却需要维护几十到几百条 Beam。此时，主机每轮组织请求、读取候选、推进状态的成本，会被短生成链路放大。GR-Scheduler 将一次搜索交给引擎持续推进：**前端提交固定成员的会话，设备保留 Beam 状态，CPU 提前准备下一阶段，终态确认资源释放后再返回结果。**

本章以 2026-10-29 拉取的 `decode_graph` 提交 [`2a598ff`](https://github.com/JiusiServe/vllm-gr/commit/2a598ffa11a25894d7d5968aad58449cdf4f1713) 为代码基线，结合尚未合并的 [PR #529](https://github.com/JiusiServe/vllm-gr/pull/529)（`393a610`）说明 vLLM 0.22.1 → 0.30.0 的升级。历史调度实验与 #530＋#537 的 L20 实验分别注明来源；它们都不是 #529 的版本升级 A/B 成绩。

## 4.1 轻量化框架：复用原生调度，缩短主机控制链路

GR 沿用 vLLM 的 Request、Scheduler、KV Cache Manager、Executor、BatchQueue 和 Future。在这些边界上附加会话、阶段、设备状态引用与回收凭证，而不再由前端逐轮拿到所有候选后重新发起下一轮搜索。

![GR-Scheduler 的职责与 B、W、Q 边界][gr04-ownership]

*图 4-1：一个执行 owner 可以包含 B 个逻辑请求；Decode 展开为 B×W 行；Q=2 表示该 owner 最多有两个阶段尚未回收。图中的分层表示职责，不要求每层都是独立进程。*

| 层次 | 负责的状态和接口 | 对低时延的作用 |
| --- | --- | --- |
| 前端 / Serving | 一次构造 `BeamBatchRequest`；渲染、分词、输出转换和客户端取消 | 将固定成员整体提交，移除前端逐步 Beam 控制循环 |
| 原生 Scheduler / KV Manager | Request 进度、token 预算、原生 KV block 和 slot mapping | 沿用同一套 token 与 KV 记账，不增加第二套 token 调度器 |
| GR 会话与阶段协议 | owner、成员顺序、dispatch 身份、设备状态依赖和 KV 租约 | 在原生调度结果上表达 Beam 依赖，限制在途工作和资源复用 |
| Executor / BatchQueue | 提交执行与采样、保存 Future、消费异步回执 | 允许下一阶段先提交，随后再消费前一阶段的回执 |
| Worker / 设备 | Beam token、累计分数、父索引、suffix KV、图和静态 buffer | 中间候选留在设备，主机只处理阶段控制信息；终态再物化输出 |

这里的“轻量化”主要指缩短热路径上的编排链路。它不意味着绕过原生 KV 分配或删除生命周期校验。对每个逻辑请求，Decode 在原生 Scheduler 中前进一个 token，Worker 内部执行 W 行；B=4、W=128 时，一次 Decode 是 **4 个逻辑 token、512 个物理 Beam 行**，而不是 512 个独立会话。

实现入口见 [`beam_batch_execution_contract.md`](https://github.com/JiusiServe/vllm-gr/blob/2a598ffa11a25894d7d5968aad58449cdf4f1713/docs/beam_batch_execution_contract.md) 和 [`gr_async_scheduler.py`](https://github.com/JiusiServe/vllm-gr/blob/2a598ffa11a25894d7d5968aad58449cdf4f1713/vllm_gr/v1/engine/gr_async_scheduler.py)。

## 4.2 并行调度：把下一阶段的 CPU 准备移入设备执行窗口

### 三种“并发”需要分别讨论

| 维度 | 含义 | 当前边界 |
| --- | --- | --- |
| B：会话内逻辑 batch | 同一会话固定包含多少条 prompt | 引擎准入支持 1～8；还需满足 token、序列数及设备容量约束 |
| W：Beam 宽度 | 每条 prompt 的物理候选行数 | CUDA 固定宽度路径要求与启动配置一致，当前上限 256 |
| Q：在途阶段数 | 同一 owner 已提交、尚未消费完成回执的 dispatch 数 | 固定为 2；不代表两条 HTTP 请求或两个会话同时执行 |
| 活跃 owner 数 | 同时占用共享执行资源的会话数 | 固定为 1；后续会话等待资源释放 |

上述约束直接来自 `GRExecutionCapacity` 与 `GR_LOOKAHEAD_DEPTH`。网络侧可以接收多个请求；设备侧仍按会话 FIFO 选择唯一 owner。会话内部的 Beam 行可批量计算，CPU 与设备也可重叠，但这不是跨会话的 continuous batching。

### N=3 的推进过程

有效生成预算 N 包括 Prefill 后产生的第一个 token。因此 N=3 的链路是 `Prefill → Decode 1 → Decode 2`。下文采用面向读者的 Decode 1/2 编号；代码中 `decode_step` 从 0 开始。

1. EngineCore 提交 Prefill 的 `execute_model` 和 `sample_tokens`。Worker 将 Forward、Beam 状态更新和控制快照依次提交到设备。
2. 原生 Request 的调度进度已经足以确定下一阶段的几何信息。只要第二个 dispatch 槽可用，EngineCore 就能提交 Decode 1，无需先取回 Prefill 的所有候选。
3. Worker CPU 提前准备 positions、offsets、prefix block 表、suffix 长度和 attention metadata。这些信息来自主机元数据，不需要读取上一阶段生成的 token。
4. 真正的 token 绑定和 Beam 状态读取在设备流中排在前一阶段采样之后。设备计算仍遵守数据依赖；提前的是主机准备和工作提交。
5. 控制回执到达后释放 dispatch 槽，再继续提交后续阶段；终态另外满足资源释放条件。

![串行回执等待与两槽异步调度的时序对比][gr04-overlap]

*图 4-2：示意时间轴，不代表实测时长。异步路径将 Decode CPU 准备移入 Prefill GPU 执行窗口；两次 GPU Forward 仍有先后依赖。准备全部完成且前驱状态就绪后，下一次设备计算才能执行。*

Graph 与异步调度在这里起到不同作用。Graph replay 降低主机提交模型算子的成本，使同一个 Worker 线程更早腾出时间准备下一阶段；BatchQueue 则允许它不等待前一阶段回执就开始这项工作。只打开 Graph、但每步仍等待回执，不能获得同样的重叠。

缓冲区复用也有明确顺序：两份 pinned staging 各自由 H2D 完成事件保护；控制快照使用独立槽和 D2H 事件。第一次 Decode 保留原生输入准备的 host fence，保护主机 buffer 复用；它不等同于等待整个 Prefill Forward 结束。机制与时序见[异步调度契约](https://github.com/JiusiServe/vllm-gr/blob/2a598ffa11a25894d7d5968aad58449cdf4f1713/docs/async_scheduling.md)。

### Prefix Cache 命中不会改变阶段语义

以 P=900、N=3 为例，P 表示进入引擎后的 prompt token 数：

| 阶段 | 无缓存命中时本轮调度 token | 已命中 896 个 prompt token 时 | 调度后累计 computed | Worker 行为 |
| --- | ---: | ---: | ---: | --- |
| Prefill | 900 | 4 | 900 | 完成剩余 prompt，生成第一个 token |
| Decode 1 | 1 | 1 | 901 | 执行 W 个 Beam 行 |
| Decode 2 | 1 | 1 | 902 | 执行 W 个 Beam 行并生成终态 |

Prefill 图按**本轮真实调度 token 数**选桶，并保持实际 attention 长度和 KV 写入掩码。例如 900 个 token 可用已捕获的 1024 桶；命中缓存后的 4 个 token 则按 4 选桶。启动时没有匹配图、元数据缺失或超出捕获容量等情况，Prefill 回退原生 Forward；请求期间不临时捕获图。Padding 可能影响 BF16 数值和 Beam 排序，历史 exact-bucket 实验不能代替当前 padded 路径的精度验收。

## 4.3 批处理、准入与背压

### HTTP 列表合批与设备容量分开配置

当前 `beam.online_batch_sizes` 默认是 `[1, 2, 4]`。在满足其他条件时，一个含 4 条 prompt 的 completions 请求组成一个 B=4 会话；含 3 条 prompt 的列表默认沿用每条 prompt 一个会话的路由，不自动拆成 B=2＋B=1，也不与另一条 HTTP 请求自动合并。

如需启用 B=3/7，可在现有 GR 配置中设置以下字段，并保留原有模型、约束表等配置：

```yaml
beam:
  execution_mode: v1
  online_batch_sizes: [1, 3, 7]
```

此时 `max_num_seqs` 至少为 7，还要满足完整 Prefill 的 token 预算。会话成员在执行前固定，阶段内必须一起被调度，不能中途加入、移除或单独推进。一个会话也只运行一个有效生成预算；预算不一致的列表保留原路由，不通过缩短某条请求的预算强行合批。

| 准入层次 | 检查内容 | 不满足时的行为 |
| --- | --- | --- |
| HTTP 合批策略 | 精确 B 是否启用、成员生成预算是否一致、初步容量 | 不接管该列表的合批路由，记录 `Beam batch declined` 原因 |
| EngineCore | Request/会话身份、运行配置、W、完整 prompt 总 token 数、原生序列数和 KV 总容量 | 不可满足的请求显式拒绝，不发布半个会话 |
| 原生 Scheduler | 当前可分配 KV、唯一 owner、在途槽 | 暂时资源不足时背压，保留等待请求和队列位置 |
| Worker / 图执行 | 成员顺序、阶段身份、B×W 行和静态 buffer、图覆盖 | 按对应 runner 的契约执行或回退；不能把缺图一概当作可静默回退 |

当前路径要求 `async_scheduling=True`、`enable_chunked_prefill=False`。完整 prompt 总长度在准入时就必须装入有效 token 预算，包括接口追加的 begin token，不能依靠 Prefix Cache 命中让超预算请求获得准入。通过准入后，实际 Prefill 可以复用已缓存 token。

原生 KV 预留的逻辑范围为每个成员 `P_i + N − 1`，通过原生 `allocate_slots` 的 lookahead 机制预留。它不代表为每个 Beam 复制完整 prompt，也不包含私有 suffix 池、Beam 状态和 CUDA Graph 的全部显存。永久容量不足与暂时无空闲 block 分开处理，避免让不可能执行的会话一直等待。

### 开启合批不等于所有 B 都有 Decode Graph

| 执行路径 | 与 B 有关的图配置 |
| --- | --- |
| CUDA ModelRunner V1 | Prefill 默认跟随在线 B 策略；Decode 仍使用单一启动形状，由 `VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE` 指定，默认 1 |
| CUDA ModelRunner V2 | Prefill 默认跟随在线 B 策略；Decode 也按策略捕获，未选择的 B 可走 eager；已成功捕获的选定 B 在运行时丢失图属于错误 |
| NPU | Decode 的精确 B 由 `beam.graph_batch_size` 决定，另有物理行容量约束；不适用 #529 的 CUDA 升级验收结论 |

扩大捕获集合会增加启动成本和图显存。应分别核对“准入支持 B”“HTTP 启用 B”和“对应 runner 的图覆盖 B”，不能只改一个列表就宣称全形状图执行。详见[批处理配置与所有权](https://github.com/JiusiServe/vllm-gr/blob/2a598ffa11a25894d7d5968aad58449cdf4f1713/docs/onerec_v1_async_quickstart.md#batching-several-prompts-in-one-http-request)。

另外，最新合入的 #525 在配置层区分了每行候选数 K 与 Beam 行数 W，但当前端到端执行仍拒绝 `K != W`，W 上限仍为 256。本章的容量和数据均不假设独立 K 已可用于生产执行。

## 4.4 生命周期：结果完成与资源可复用是两个条件

单 owner 能简化共享图和静态 buffer 的使用，但安全切换 owner 的前提是上一会话已实际释放资源。否则，取消后的迟到回执、仍在执行的 GPU 工作或尚未完成的输出转换，都可能碰到下一会话正在复用的状态。

![GR 会话的资源状态与回收确认][gr04-retirement]

*图 4-3：资源生命周期与业务成功/失败状态分别维护。正常末次输出可携带释放证明，省去独立 retirement 往返；需要独立清理时，必须经过零 token 控制帧和匹配确认。*

| 场景 | 处理方式 | 保证的边界 |
| --- | --- | --- |
| 正常结束 | 最后输出包含匹配的 `resources_released=True`，或待独立回收确认后交付 | 终态结果至多交付一次，随后才允许下一 owner 使用资源 |
| 等待中取消 | 从会话和原生请求状态退出；尚未分配 Worker 资源时可直接释放 | 不为已取消的等待请求启动设备执行 |
| 执行中取消 | 停止后继推进、排空在途 dispatch、清理 Worker 状态 | 不发送取消后的迟到成功结果，不提前复用 buffer |
| 重复或旧回收回执 | 比对 `session_id / owner_generation / retire_id` | 旧 owner 的回执不能释放当前 owner；缺失或不匹配的有效回收证明触发协议错误 |
| 阶段失败 / Worker 故障 | 请求转换错误进入失败路径；致命 Worker/CUDA 错误交给原生 Executor 故障处理 | 不把缺失回执解释为阶段已成功完成 |
| 暂停或队列无新请求 | 仍把清理、回收和终态交付视为待完成工作 | 最后一个请求的清理不能因为没有新推理工作而停住 |

在途阶段还持有原生 KV block 的引用计数租约。取消原生 Request 并不能立即让 GPU 仍在读取的 block 被重新分配；相应阶段回执被消费后，调度器才归还租约。Worker 中保存一个 Python block 对象引用不能替代这层保护。

这些机制提供的是引擎内部的身份校验、至多一次终态交付和资源隔离；当前会拒绝重复的活跃会话 ID，不提供跨网络重试的持久化结果去重服务。代码见 [`consume_gr_retirement`、`update_gr_from_output` 和 `abort_gr_session`](https://github.com/JiusiServe/vllm-gr/blob/2a598ffa11a25894d7d5968aad58449cdf4f1713/vllm_gr/v1/engine/gr_async_scheduler.py)。

## 4.5 数据：调度收益与设备计算成本分别量化

### 四组消融：Graph 与异步调度配合降低服务时延

已有实验采用 NVIDIA L20、OneRec-1.7B、BF16、B=1、P=1024、W=128。四组固定 Decode Graph、输入、生成预算、executor 和缓存条件，仅对比 Prefill eager/graph 与串行/异步提交。每组独立运行 3 次，每次预热 10 个请求、测量 100 个请求。

**统计口径是“每次运行的服务时延中位数，再对 3 次运行取均值”**，终点为 Worker 资源释放证明；不是 300 个样本的合并 P50，也不是客户端 HTTP E2E 或 P99。

| Prefill / 调度方式 | 服务时延（ms） | 相比 eager＋串行减少（ms） | 相对下降 |
| --- | ---: | ---: | ---: |
| Eager＋串行 | 62.488 | — | — |
| Eager＋异步 | 59.270 | 3.218 | 5.15% |
| Graph＋串行 | 62.212 | 0.276 | 0.44% |
| Graph＋异步 | 58.598 | 3.890 | 6.23% |

![四组调度消融实验的服务时延][gr04-ablation]

*图 4-4：横轴从 0 开始，越短越好；百分比相对 eager＋串行计算。该图重绘已有测量，没有执行新 benchmark。*

固定 Graph 后，异步相对串行减少 3.614 ms，下降 **5.81%**。五份历史 trace 均观察到 D1 的 CPU 准备、metadata 构建和绑定落在 Prefill GPU 执行窗口内；当时原生 BatchQueue 已能在消费 Prefill 回执之前提交下一阶段，无需为此新增调度线程。

这些结果来自[上游保留的历史验证记录](https://github.com/JiusiServe/vllm-gr/blob/2a598ffa11a25894d7d5968aad58449cdf4f1713/docs/async_scheduling_validation.md#historical-graph-coverage-and-scheduling-results)，早于后续清理和 padded Prefill 调度变更。记录中的历史基础为 `6bca38c`，当前仓库保留的是实验摘要，原始逐请求记录位于当时的本地归档。因此这里只报告该工作负载的观察结果，不给出置信区间，不外推到当前全部 B、P、W 或 #529 的性能。

同一历史记录还列出了 100 条原始长度输入的 SID pass@1 / pass@32 / recall@32 为 `0.06 / 0.23 / 0.03279`，并通过当时的既有门禁。旧的 exact-bucket 策略只命中其中 2/100 个 Prefill；人为控制到 1K 的输入命中 100/100。后者改变了上下文，不能代替原始输入的质量结果；这些覆盖率也不能当作当前 padded 图路径的命中率。

### 六组 L20 数据：批量并行仍受设备计算形状约束

2026-10-28 的另一组实验比较 #530＋#537 组合提交 `145c6045a4dd26385c72b7042d55c9abc2701fd9` 与 SID-GR。它没有开关异步调度，**用于展示设备侧成本随 L/B/W 的变化，不用于归因调度收益**。

共同条件为同一张 L20（SM89、92 SM）、OneRec-1.7B、BF16 权重与 KV、TP=1、全词表无约束 Beam、K=W、N=3，跨请求 Prefix Cache 关闭。GR 的 Prefill/Decode 均显式使用 FA2；SID-GR 使用 recsys-examples v26.08 的 `97062d97`，在该卡分派到 SM80 CuTe decode 核。输入为固定种子 `20261028` 的合成 SID 历史，逐批核对双方输入 hash，未评估真实业务 Recall。

执行顺序为 GR A1 → SID B1 → SID B2 → GR A2。每个形状、每次启动预热 12 批、计时 48 批，因此每个后端每个形状有 96 批正常 API 样本；另有 6 批 profiler 样本。下表 attention 和模型列来自 profiler，单位为**每批、28 层、两次 Decode 的 GPU kernel 时间累计（ms）**。模型列排除 LM-head、Top-k 和图外状态准备，不包含 GPU 空隙。

| L / B / W | GR attention | SID attention | GR/SID attention | GR Decode 模型 | SID Decode 模型 | GR 正常 API 均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 4 / 256 | 3.776 | 4.354 | 0.87× | 60.911 | 61.912 | 198.120 |
| 1024 / 4 / 256 | 6.438 | 7.363 | 0.87× | 63.527 | 64.865 | 266.884 |
| 8192 / 1 / 32 | 14.825 | 5.346 | 2.77× | 24.959 | 15.884 | 359.398 |
| 10240 / 1 / 32 | 18.445 | 6.654 | 2.77× | 28.546 | 17.105 | 467.950 |
| 10240 / 1 / 256 | 18.554 | 13.400 | 1.38× | 34.424 | 29.218 | 487.563 |
| 10240 / 4 / 256 | 55.471 | 61.995 | 0.89× | 112.366 | 119.537 | 1901.869 |

正常 API 列来自单独的无 profiler 测量，包含 Prefill、两步 Decode、Beam 选择、候选 token/文本/分数物化及同步，是每批调用耗时。它与 kernel sum 计时范围不同，不能相减得到“CPU 调度耗时”，也不能用 B=4 的总耗时直接当成单条请求时延。

![GR 六个形状的 Decode 模型计算构成][gr04-device-cost]

*图 4-5：GR Decode 模型 kernel sum 的组成；每行右侧标出该形状的模型总时间。占比根据已有分项时间计算，分项与总量存在原始报告的小数舍入误差。*

数据对调度设计给出两个约束：

- **小 B、窄 W、长上下文时，主机重叠之外还存在设备并行度问题。** 10K/B1/W32 的 GR prefix 主核只有 16 个 CTA，而 L20 有 92 个 SM；SID 分段主核为 80 个 CTA。GR attention 占模型 kernel sum 约 64.6%，且用时是 SID 的 2.77 倍。这支持进一步验证分段 KV 并行，但两家 KV 布局也不同，不能把差距全部归因于 CTA 数。
- **较大 B/W 下，attention 更快也不保证整个 API 更快。** 512/B4/W256 的 GR attention 比 SID 少约 13.3%，但线性层为 53.824 ms，占 GR 模型 kernel sum 约 88.4%；GR/SID 正常 API 均值分别为 198.120/196.877 ms。合批、主机重叠、选择算子与模型计算需要按各自计时边界评价。

本地原始记录为 `vllm-gr-l20-decode-530-537/results/l20-decode/paired/` 下的 `l20_decode_comparison_zh.md`、`decode_analysis.json`、`provenance.json` 和四臂逐点 JSON。环境为 Torch 2.11.0+cu130、Transformers 4.57.6、CuTe DSL 4.5.2、quack 0.5.0；后两项与 SID kernels extras 的固定安装组合不同，结论限定本机实际环境。本章附带[作图数据与来源说明][gr04-data]，保留两组实验的提交、条件和表格数值；它是摘要，不替代原始 trace。

## 4.6 PR #529：保留调度语义，迁移 vLLM 0.30 的接口契约

#529 的主要工作是把上述路径接入 vLLM 0.30.0 的运行时。它不把 Q 从 2 扩大，也不把活跃 owner 从 1 改成多会话；公开 API 名称 `beam_search_v1()` 中的 V1 是 GR 协议名称，不代表必须使用 ModelRunner V1。

| 升级边界 | #529 的具体改动 | 对调度正确性的意义 |
| --- | --- | --- |
| 启动与 Serving | CUDA 依赖及版本检查切换到 0.30.0；迁移到 `GenerateBaseServing` / Renderer 等新入口 | 先校验运行时，再安装对应补丁；普通生成继续沿原生路径 |
| EngineCore 握手与传输 | 调用原生 `_make_ready_response()`；保留多模态 cache-miss 重试及故障容错 utility 分发 | 跟随完整 READY 配置与控制协议，避免手写旧字段导致启动或控制消息不兼容；不因此开放 GR 多模态能力 |
| 原生 Scheduler | 在嵌套 wrapper 中传递 `schedule(throttle_prefills=...)`；处理 `finish_requests` 返回的 Request 对象 | 不丢失原生 Prefill 节流参数，正确同步原生完成/取消与 GR 会话状态 |
| KV block 租约 | 保存每个 KV group 的租约，按组调用 `free_blocks` | 对齐新的 block pool 释放与驱逐顺序；不把多组展开成一次释放 |
| GPU 输入准备 | 保留 `_prepare_inputs` 三项返回值，包括最大采样宽度；适配统一多维 RoPE 判断和原生 PP 等待条件 | 快路径不能丢失原生输入契约，不适用时交回原生路径 |
| Attention / logits | 使用原生 per-layer KV view 与 layout 协商；传递 `skip_gather`、保留 `head_dtype` | 调度传入的 KV 和 logits 仍遵守 0.30 的存储、投影与收集语义 |
| Prefill Graph | 接受原生已经 padding 的输入；GR 仍按真实 scheduled token 前缀选桶和拷贝 | 不把原生 padding 长度误当成有效 token 数，也不因这层 padding 无谓回退 |
| HTTP session hint | 通过原生 `_get_session_id()` 解析提示，向合批和 Beam 入口传递 | 保留 body/header/extra 参数的原生优先级，不把路由提示混同于请求 ID 或执行 owner |
| 关闭流程 | 同步引擎向 EngineCore client 传递配置的 `shutdown_timeout` | 父进程按配置等待排空，避免默认清理窗口提前结束；超时为 0 仍保留 abort 模式的尽力清理 |

这些改动可在 [#529 固定提交的代码](https://github.com/JiusiServe/vllm-gr/tree/393a610182cf5f5c22ba5163492413adb1f19b1b) 中复核。其中 `throttle_prefills` 是对原生控制参数的透传，不能据此宣称 GR 已新增跨会话的 Prefill/Decode 优先级调度算法。

### ModelRunner V2 与数值一致性

ModelRunner V2 仍为显式选择的实验路径，默认 runner 仍是 V1。#529 接入原生 `BatchReqState`、请求顺序、padding 和 `postprocess_num_computed_tokens`，消费并清理原生 pending forward 状态后再生成 GR 输出。

V2 图显存估算也计入 GR Prefill 捕获计划、Decode 图和固定状态。由于不同 token/request 形状的成本不均匀，使用原生临时池测量完整计划，并清理临时图和 owner；这会增加启动工作量。仅在上游启用相应图显存 profiling 时生效，不能把估算值当成服务性能结果。细节见 [V2 升级说明](https://github.com/JiusiServe/vllm-gr/blob/393a610182cf5f5c22ba5163492413adb1f19b1b/docs/modelrunner_v2_multibatch.md#vllm-030-integration)。

独立引擎对比还暴露了一个与调度无关但会影响结果验收的问题：Inductor 为 Q/K norm 和 RoPE 生成的归约核可能选择 64 或 128 元素的归约块，差异在 Prefill 阶段已经出现。#529 对 CUDA Beam ModelRunner V1＋Inductor 默认设置 `dynamic_scale_rblock=false`，保留显式配置的优先级，将该选择纳入原生编译缓存身份。

该策略针对已观察到的独立引擎差异，不保证不同 B、GPU 或编译器版本下结果完全相同；ModelRunner V2、NPU 和非 Beam 引擎保留各自默认值。业务 Recall 与性能仍需针对升级后的候选版本独立验收，见[归约策略说明](https://github.com/JiusiServe/vllm-gr/blob/393a610182cf5f5c22ba5163492413adb1f19b1b/docs/async_scheduling_validation.md#independent-engine-reduction-policy)。

### 升级验证覆盖到了哪里

下表整理 #529 当前代码中的验证矩阵与断言，不代表本章编写时重新执行过这些测试，也不等同于完整业务上线验收：

| 验证项 | 代码中的范围 | 要验证的结果 |
| --- | --- | --- |
| 公共生命周期 | V1/V2 runner × 离线/在线，共 4 种配置 | 真实模型阶段推进、取消后恢复、终态与进程清理 |
| HTTP completions | V1/V2 × 默认有约束 / 自定义 B 有约束 / 默认无约束，共 6 种配置 | 默认 B=1/2/4、自定义 B=1/3/7；独立离线/HTTP 引擎同容量配对、候选顺序与分数、并发会话、等待/执行中 TCP 断连和恢复 |
| 离线无约束生成 | V1/V2 × Eager / 默认 Graph / 显式 FULL Prefill，共 6 种配置；每种处理 B=1/4、W=128 | 实际 Decode 和 Prefill replay；FULL 用例含 1024 及 4102→4224 padding；不以“开启 Graph”代替命中证据 |
| 资源释放 | 上述真实模型生命周期用例 | 无残留原生请求或 Worker 会话，EngineCore 进程确实退出且 exit code 为 0 |

测试定义见 [`test_onerec_v1_scheduler_st.py`](https://github.com/JiusiServe/vllm-gr/blob/393a610182cf5f5c22ba5163492413adb1f19b1b/tests/st/test_onerec_v1_scheduler_st.py) 和 [`test_offline_beam_search.py`](https://github.com/JiusiServe/vllm-gr/blob/393a610182cf5f5c22ba5163492413adb1f19b1b/tests/test_offline_beam_search.py)。PR 提交说明记录了阶段性通过结果，但 PR 正文尚未给出完整升级性能表，不能将其中某次回归数量当作最终 head 的全量验证结果。

**NPU 升级仍未放开。** #529 的环境检查明确阻止尚未验证的 vLLM 0.30.0 / vllm-ascend 0.30.0rc2 路径，vendor runtime 和 GR NPU 路径仍待验证。它与 `decode_graph` 中已有的 NPU 实现是两个版本状态，不能相互替代。

## 4.7 落到部署和评估的指标

部署时应记录精确提交、runtime、runner、B/W/N、图捕获集合、缓存模式和输出物化范围，再分别评价以下指标：

| 指标 | 应说明的边界 | 本章已有证据 |
| --- | --- | --- |
| 服务内调度时延 | 明确服务内计时起点、是否含排队；历史实验终点为 Worker 资源释放证明 | 历史四组实验；Graph＋异步为 58.598 ms 的“各次中位数均值” |
| 正常 API / HTTP E2E | Prefill、Decode、输出物化、同步；HTTP 还含传输、Serving 和排队 | L20 表为正常 API，未给出在线并发 P99 或 QPS |
| 设备计算 | profiler 中明确归属的 Forward / attention / 选择算子 | 六个 L/B/W 形状；kernel sum 不等于墙钟时延 |
| 正确性与清理 | 候选及分数、真实数据 Recall、取消恢复、图命中、资源释放、进程退出 | 历史质量摘要与 #529 验证矩阵；不能替代升级后业务验收 |

当前 V1 不再自动暴露旧前端的 `vllm_gr_beam_search_*` 阶段计时指标；原生 `/metrics` 也不等价于 GR Beam 分阶段耗时，缺失指标应记为不可用。避免为了恢复旧计时方式在热路径强行同步设备，详见[指标边界](https://github.com/JiusiServe/vllm-gr/blob/2a598ffa11a25894d7d5968aad58449cdf4f1713/docs/beam_search_metrics.md)。

对当前实现，调度优化的确定性落点是：一次会话提交、设备持久状态、两槽阶段流水和可核验的资源交接。后续若扩大批量或引入更多并发 owner，需要同时证明计算收益、排队尾延迟和资源隔离；已有的 Q=2 机制与功能覆盖不能直接推出这些收益。

<!-- Markdown 预览采用首组相对链接；Jekyll 经 Liquid 处理后只保留带 baseurl 的第二组。 -->
<!-- {% unless site %} -->
[gr04-ownership]: ../../assets/figures/vllm-gr/04-scheduler-ownership.svg
[gr04-overlap]: ../../assets/figures/vllm-gr/04-scheduler-overlap.svg
[gr04-retirement]: ../../assets/figures/vllm-gr/04-scheduler-retirement.svg
[gr04-ablation]: ../../assets/figures/vllm-gr/04-scheduler-ablation.svg
[gr04-device-cost]: ../../assets/figures/vllm-gr/04-scheduler-device-cost.svg
[gr04-data]: ../../assets/figures/vllm-gr/04-scheduler-data.json
<!-- {% else %} -->
[gr04-ownership]: <{{ '/assets/figures/vllm-gr/04-scheduler-ownership.svg' | relative_url }}>
[gr04-overlap]: <{{ '/assets/figures/vllm-gr/04-scheduler-overlap.svg' | relative_url }}>
[gr04-retirement]: <{{ '/assets/figures/vllm-gr/04-scheduler-retirement.svg' | relative_url }}>
[gr04-ablation]: <{{ '/assets/figures/vllm-gr/04-scheduler-ablation.svg' | relative_url }}>
[gr04-device-cost]: <{{ '/assets/figures/vllm-gr/04-scheduler-device-cost.svg' | relative_url }}>
[gr04-data]: <{{ '/assets/figures/vllm-gr/04-scheduler-data.json' | relative_url }}>
<!-- {% endunless %} -->
