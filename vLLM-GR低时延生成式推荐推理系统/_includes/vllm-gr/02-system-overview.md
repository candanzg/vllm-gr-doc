# 2 低时延生成式推荐推理系统 vLLM-GR

vLLM-GR 是一个基于 vLLM 的生成式推荐推理引擎，当前聚焦 OneRec 类模型的 SID 候选生成路径，同时保留普通 LLM 的 `generate()` 直通能力。二者是两条独立路径，能力不可相互外推。

本章先给出一次搜索的计算全景，再说明系统由哪些层组成、各自复用上游还是新增了什么。

## 2.1 整图计算流

以有效预算 N=3、固定宽度 W 为例，一次完整的推荐搜索如下。

```
前端提交（一次）
    │
    ├─ Prefill：处理用户上下文，取最后一行 logits
    │          约束 TopK 选出第一段 SID → 初始化设备 Beam 状态
    │
    ├─ Decode #1：B×W 行展开 → forward → 候选选择 → 第二段
    │            （按父索引重排后缀 KV）
    │
    └─ Decode #2：同上 → 第三段 → 设备端排序打包
                 │
                 └─ 单份 D2H → 前端构造 BeamSearchOutput
```

需要强调三点：

**其一，N=3 不是三次 Decode。** Prefill 已经在最后位置选出第一个 token，因此只有两次 Decode forward。如果接口还包含 begin/end 标记，它们与有效 SID token 的预算关系需要单独处理，不能把所有返回标记都算成设备执行的 Decode 步数。

**其二，中间状态不回主机。** 每一轮 Decode 结束后，真正获胜的 token、累计分数与父关系都留在设备侧。回传的只是紧凑控制信息（阶段身份、几何、完成状态），终态才做一次设备端排序打包和单次 D2H。

**其三，选择发生在同一个 item 内部。** B 个逻辑请求展开为 B×W 个物理执行行，但候选只在各自 item 内竞争；第二个用户不会因为第一个用户分数更高而丢失全部 Beam。父索引也在各 item 内解释。

![从提交到终态回收的请求时序](/assets/figures/vllm-gr/03-request-lifecycle.png)

*图 3：同一会话中的 Prefill、两次 Decode 与终态回收。阶段回执是控制信息；正常最后阶段可以合并终态候选与释放证明。图中责任主体是逻辑角色，实际进程部署由 Executor 等配置决定。*

这条链路相对上游 Beam 的关键变化，在于**调用边界**：

| 环节 | upstream vLLM | vLLM-GR |
| --- | --- | --- |
| 引擎请求数 | `1 + (N−1)·W` | 1（一次会话） |
| Beam 状态位置 | Python 前端对象 | Worker / 设备 |
| 前缀计算 | 每步重新 Prefill | 一次 Prefill + 后缀 |
| 中间结果 | top-2W logprobs 全量回主机 | 紧凑控制信息 |
| 候选选择 | 前端 Python 排序 | 设备端算子 |

## 2.2 总体架构

系统可以按"复用 / 接入 / 新增"三类职责理解。这个划分很重要：它解释了哪些行为可以按 vLLM 的文档来理解，哪些必须查 GR 自己的契约。

![vLLM-GR 运行时分层架构：上游复用、接入改造与新增机制](/assets/figures/vllm-gr/03-runtime-architecture.png)

*图 4：从 EntryPoints、Engine 到 Model / Ops 的逻辑职责图。蓝色表示复用 vLLM，黄色表示在上游职责处接入，粉色表示 GR 新增机制。顶部 plugin 在启动时安装扩展；图中分组不代表独立进程，也不要求存在同名 Python 类。*

### 复用层（蓝色）

模型权重加载、原生 Prefill、分页 KV 管理、调度记账、请求与响应协议都走上游实现。GR 不重写这些能力，因此上游在模型支持、量化方案、并行策略上的演进大部分可以直接继承。

### 接入层（黄色）

通过 `vllm.general_plugins` 入口注册，vLLM 在 **main / EngineCore 子进程 / Worker 子进程**中各自执行一次，所以初始化必须幂等。启动链路是：

```text
插件入口 → ModelRunner 默认 → 版本闸门 → 补丁链 → 约束表加载 → 图捕获
```

两处细节值得单独说明：

- **ModelRunner 默认值。** vLLM 0.22 会对部分模型（如 Qwen3）自动启用 ModelRunner V2，绕过 GR 的 V1 worker 补丁并可能**静默产生错误的 Beam 结果**。GR 在配置构建前将未设置的开关置为 0；显式设置 V2 仍作为实验路径保留。
- **版本闸门。** 补丁针对上游私有符号，因此启动时校验 vLLM 版本，不匹配直接拒绝启动。版本号一致并不等于私有接口兼容——GR 文档记录的基线与本机 HEAD 就不是同一提交。

每个运行时补丁都必须登记在补丁清单中（owner、安装器、目标符号、进程角色、风险、移除条件），并配有契约测试。补丁是否已安装、目标方法是否符合预期，都会影响引擎能否进入可服务状态。

### 新增层（粉色）

- **会话提交与阶段 metadata**：前端一次提交，EngineCore 完成准入与阶段推进，调度器不会看到半个批次。
- **Worker Beam 决策**：候选展开、累计分数选择、父索引重排。
- **私有后缀 KV 与 Attention 合并**：共享 prompt 走原生分页 KV，分叉后的短历史进 GR 自己的池。

![共享 prompt 与私有 suffix 的 Attention](/assets/figures/vllm-gr/02-prefix-suffix-kv.png)

*图 5：同一个 item 的多条查询读取一份分页 prompt 和各自的短 suffix，分别计算 Attention 后用 LSE 合并。箭头表示读取与计算依赖，区域大小不表示内存比例。*

图中三类状态是分开放的：原生分页 KV 保存共享 prompt，设备 Beam 状态保存候选 token、分数与父关系，GR 私有 suffix KV 保存分叉后的历史。它们通过阶段 metadata、父重排和资源回收协议协作。

## 2.3 当前能力边界

以下是代码层面的准入约束，不是全部设备上的实测结果。

| 项目 | 当前范围 |
| --- | --- |
| 请求形态 | 调用方组成固定成员 batch，返回最终候选集合 |
| 批次与宽度 | B=1..8，W=1..256，请求需匹配引擎配置 |
| 生成预算 | 固定有效预算；canonical 约束路径 N≤3 |
| 执行 | 单执行 owner，Q=2 在途 dispatch |
| 停止 | 要求 `ignore_eos=True`，按长度结束 |
| 并行 | TP/PP/DP/DCP/PCP 均为 1 |
| Prefill | 需完整完成剩余 prompt，拒绝 chunked / partial |
| 功能组合 | 拒绝量化、LoRA、多模态、hybrid、speculative、KV/EC transfer、microbatching |
| 平台 | CUDA 主线；NPU 有独立适配，CUDA ModelRunner V2 为实验性 |

固定 batch 不等于跨请求自动合批。在线 completions 可以把策略允许的 prompt 列表组成一个会话，但多个独立 HTTP 请求自动加入同一执行 batch 属于另一层准入设计，当前不做。

这些边界属于 GR 持久 Beam 路径，**不能外推为整个 vLLM 的限制**。普通 `generate()` 直通路径不受这些约束。

## 2.4 本章小结

vLLM-GR 的核心取舍可以概括为一句话：**用设备端持久状态换掉每步 W 次主机往返**。

这样做的收益取决于原本主机编排占多少时间，代价则是一组更紧的准入约束和明确的版本绑定关系。后续三章分别展开这三块：

- 第 3 章讨论推理引擎本身：Beam Search 优化、整图优化、候选集优化；
- 第 4 章讨论调度与并发：轻量化框架与并行调度；
- 第 5 章讨论池化加速：两级 KV 组织，以及 GPU / NPU 共用的后缀池契约。
