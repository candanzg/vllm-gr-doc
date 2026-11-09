# 5 池化加速 GR-Cache

生成式推荐的 KV 有一个鲜明的特点：W 条 beam 共用同一段较长的用户上下文，真正分叉的只有最后 N 个 token。vLLM-GR 据此把 KV 拆成两级管理。prompt KV 留在 vLLM 的分页池里，跨请求复用直接交给上游的自动前缀缓存（APC）；分叉后的后缀 KV 放进 GR 新增的稠密后缀池 BeamAttentionPool，只在一次搜索内有效。每次 Decode attention 都会同时读取这两级。

把共享前缀和私有后缀分开计算，这一思路参考了 Hydragen 和 xGR；后缀池的布局和 NPU 算子签名则与 xllm-ops 保持一致。本章着重介绍它在 vLLM 插件框架中的落地方式，以及在 L20 上实测的显存和延迟。

文中的显存数据取自 vLLM-GR `decode_graph` 分支 `33a1fdb`，测试环境为单张 L20，模型为 OneRec-1.7B。

## 5.1 GR 负载下的 KV 特征

![KV 显存布局：共享 prompt + 私有短后缀][gr05-layout]

*图 5-1：W=128 时单个 item 的 KV 占用，以及 vLLM-GR 在启动时预分配的显存。示意部分不按比例绘制；表中标注“估算”的一行未经实测。*

OneRec-1.7B 有 28 层、8 个 KV head，head_dim 为 128，使用 bf16。一个 token 在所有层上的 KV 合计 112 KiB（实测 114,688 B），1024 个 token 的 prompt 就是 112 MiB。下面取 N=3，比较 W=128 条路径在不同组织方式下的显存占用（图 5-1）。

最直接的做法是让每条 beam 各自持有完整的 KV。这里只把它当作完全不共享时的参照，并不对应哪个具体系统。128 条 beam、每条 1027 个 token，合计约 14 GiB；而 L20 在 gpu_memory_utilization=0.5 时分给 vLLM 的分页池只有 16.4 GiB，只够放下一个 item。

vLLM 的 APC 以 16 token 为一块做前缀共享。1024 token 的 prompt 正好 64 块，可以全部共享。问题在于每条 beam 新生成的 token 必须写进自己的私有块，哪怕只用到 3 个位置，也要占满 16 个。按此估算约为 336 MiB，其中 224 MiB 花在了给这 3 个 token 预留的块上。这一行是计算值，没有实测。

在 vLLM-GR 中，prompt 仍由 vLLM 分页管理，只存一份；分叉后的后缀写进 BeamAttentionPool，每条 beam 按 `max_decode_steps` 预留位置，不再按块对齐。单个 item 一共 154 MiB，约为完全不共享时的 1/93，比块粒度共享也少了一半以上。

## 5.2 两级 KV 组织

![两级 KV 组织与 GPU / NPU 同一契约][gr05-two-level]

*图 5-2：第一级复用 vLLM 分页池，第二级为 GR 新增的稠密后缀池；下方表格列出后缀池三类操作在两个平台上的实现。*

第一级是 prompt KV。GR 没有改动分页池，块的分配、回收和淘汰都沿用上游逻辑。每个 item 的 prompt 只存一份，供 W 条 beam 共同读取；相同的上下文再次到来时，按块哈希命中（见 5.4）。由于这一级完全使用上游实现，vLLM 后续在分页管理和前缀缓存上的改进，GR 都可以直接受益。

第二级是后缀 KV。BeamAttentionPool 为每一层分配一对 K/V 张量，形状为 `[max_batch, W, H_kv, S, head_dim]`，其中 H_kv 是 KV head 数，S 是 `max_decode_steps`。整个池在启动时一次分配完成，运行期间不再申请或释放；每次搜索从 free-list 领取一个 slot，结束后归还。

后缀单独成池，主要出于三点考虑。其一是粒度：后缀只有几个 token，按 16 token 分块时块内大半闲置，按实际步数分配则没有块内碎片。其二是地址固定：池的地址在运行期间不变，CUDA Graph 捕获的指针在回放时始终有效，Decode 阶段因此可以整图回放（见 4.2）。其三是重排简单：每步按父索引重排后缀，只需在稠密张量上做一次 gather/scatter，不涉及分页块的分配和引用计数（见 5.3）。

两级 KV 在 Decode attention 中汇合：prefix 和 suffix 各自算出局部输出与 log-sum-exp，再按 LSE 加权合并（见第 2 章图 5）。

后缀池上的操作只有三类：写入本步 K/V、两级 attention、按父索引重排。GPU 和 NPU 共用同一份池布局 `[slot, beam, kv_head, step, head_dim]`，各自提供算子实现。在 CUDA 上，写入由 Triton kernel 完成，步号从设备端的标量缓冲读取，因此可以被图捕获；attention 的 prefix 部分使用 FlashAttention，后缀计算与 LSE 合并融合在一个 Triton kernel 中；重排采用全层 gather/scatter。在 NPU 上，则直接调用 xllm-ops 的 `cache_unshared_kv`、`x_attention` 和 `select_unshared_kv`。两个平台还共用一份 Torch 参考实现做数值对照。这样，上层的调度、决策和结果打包逻辑都不必区分底层平台。

## 5.3 后缀池的生命周期与父索引重排

![后缀池的生命周期与父索引重排][gr05-reorder]

*图 5-3：上半部分为一次搜索中 slot 的生命周期，下半部分以 W=4 为例，说明原地拷贝产生的先写后读错误，以及 vLLM-GR 的两段式重排。*

以 N=3 为例，一次搜索包括一次 Prefill 和两次 Decode。Prefill 把 prompt 写入分页池，同时从 free-list 取得一个 slot。此后每次 Decode 先把本步 K/V 写进 slot 的对应列，再执行两级 attention 和候选选择；选择完成后按父索引重排 slot 中的各行，保证第 i 行始终是第 i 条存活 beam 的历史。最后一步之后不会再读取，因此不做重排。搜索结束时，结果在设备端排序打包，只做一次 D2H，slot 随即归还。

slot 归还时并不清零。写入是对指定位置的精确覆盖，attention 只读到当前步，新会话又从第 0 步重新写起，因此旧数据不会被读到。

重排的难点在于父索引通常有重复，还可能乱序。以图 5-3 中的 parent = [1, 1, 0, 3] 为例，如果直接在原地逐行执行 `pool[i] ← pool[parent[i]]`，第 0 行会先被覆盖，等到第 2 行读取第 0 行时，拿到的已经是改写后的数据，结果取决于拷贝顺序。

vLLM-GR 在 CUDA 上把重排拆成两段：先 gather，按父索引把各行读入 scratch，此时池尚未被修改；再 scatter，把 scratch 整体写回池。这样无论父索引如何重复、如何排列，结果都相同，并且按位拷贝、逐位一致。两段各执行一次 kernel launch，覆盖全部 28 层。各层池的基地址在启动时写入一张固定的指针表，kernel 直接按地址寻址，无须每步重建索引张量。

代价是一块常驻的 scratch，形状为 `[L, W, H_kv, S, head_dim]`，OneRec-1.7B 在 W=128 时占 42 MiB。NPU 侧调用 xllm-ops 的 `select_unshared_kv` 原地完成重排，xGR 论文则用方向索引规避同类冲突。我们选择以这块固定显存换取更简单的实现：结果不依赖父索引的分布，整段流程也能被 CUDA Graph 捕获。

## 5.4 跨请求前缀复用

![跨请求前缀复用：cache-miss 与 cache-hit][gr05-hit-miss]

*图 5-4：上半部分是 Prefill 实际计算的 token，下半部分是 100 对样本的端到端延迟全样本。*

第一级 KV 沿用上游 APC。启用 APC 后，同一个 prompt 再次提交时，已缓存的块直接从分页池读取，Prefill 只计算未命中的部分。在可复用的完整前缀块均命中的情况下，实际调度的 token 数为 `((L − 1) mod block_size) + 1`，其中 L 为经过输入处理、包含额外标记的有效 prompt token 数；末尾仍需执行模型前向，以获得首个 SID token 的 logits。若仅部分前缀命中，剩余计算量还取决于实际命中长度。

> 本段补充：[Xinming Han（@Lechmere-1）](https://github.com/Lechmere-1)

在本文讨论的稠密 Attention 路径中，未显式指定 `block_size` 时，GPU 默认每块 16 个 token，NPU 默认每块 128 个 token。本节 L20 实验采用 GPU 的 16-token 块配置，因此输入长度为 1024 时，缓存命中后的 Prefill 需计算 16 个 token；若采用 NPU 的默认 128-token 块配置，相同长度的 prompt 在上述命中条件下需计算 128 个 token。具体块大小以引擎实际配置为准。

图 5-4 的数据来自离线基准测试：每个样本先提交一次（cache-miss），紧接着再提交同一个 prompt（cache-hit），共 100 对。

| | p50 | p90 | p99 |
|---|---|---|---|
| cache-miss | 52.08 ms | 52.13 ms | 52.97 ms |
| cache-hit | 33.01 ms | 33.28 ms | 33.95 ms |

命中后 p50 降低 19.1 ms，约 37%。两组样本都很集中，miss 和 hit 的 p50 与 p99 都只差约 0.9 ms。

这里有一个需要留意的配置项：prefill 图的 KV 上限。cache-hit 的 Prefill 虽然只调度 16 个 token，读取的却是完整长度的上下文 KV。如果环境变量 `VLLM_GR_PREFILL_GRAPH_KV_BOUND` 小于完整 prompt 长度，这一步会静默回落到 eager 执行。对照实验中，这会使 hit 的 p50 增加约 23 ms，命中的收益几乎被抵消；而 miss 侧只受约 0.3% 的影响，只看 miss 的数据很容易忽略这个问题。该变量在每个 Worker 进程捕获图时读取一次，应设为不小于 prompt 长度的值，例如直接取 `max_model_len`。

## 5.5 显存实测与代价

后缀池和 scratch 的大小完全由配置决定：

```text
后缀池 = 2(K/V) × L × max_batch × W × H_kv × S × head_dim × dtype 字节
scratch = 2(K/V) × L × W × H_kv × S × head_dim × dtype 字节
```

以 max_batch=8、W=128、S=3 为例，后缀池为 336 MiB，scratch 为 42 MiB，加上后缀拷贝缓冲、决策状态和后处理工作区共 16.4 MiB，GR 新增显存约 394 MiB，占同卡分页池（16,819 MiB）的 2.3%。这些缓冲在第一个请求到来之前就已分配完毕，执行请求也不会改变它们的大小。

![显存扫描：GR 预分配随 W 线性增长，与输入长度无关][gr05-sweep]

*图 5-5：W ∈ {64, 128, 256} × 输入 ∈ {512, 1024, 2048} 共 9 组，每组单独启动引擎，在首个请求后读取 Worker 上的张量与调度器块池。*

为确认这些数字并非单一配置下的结果，我们在 L20 上让 W 和输入长度各取三档，共扫描了 9 组。

| W | GR 新增合计 | 其中后缀池 | 其中 scratch | 其他缓冲 | 占分页池 |
|---|---|---|---|---|---|
| 64 | 196.0 MiB | 168 MiB | 21.0 MiB | 7.0 MiB | 1.2% |
| 128 | 394.4 MiB | 336 MiB | 42.0 MiB | 16.4 MiB | 2.3% |
| 256 | 801.3 MiB | 672 MiB | 84.0 MiB | 45.3 MiB | 4.8% |

从扫描结果可以看出三点。首先，GR 新增显存与输入长度无关：同一 W 下，输入从 512 增加到 2048，读数完全相同，因为后缀池只与 N 有关，prompt 都留在分页池中。其次，后缀池和 scratch 随 W 严格线性增长，与上面的公式一致；其他缓冲增长得稍快，W=256 时达到 45.3 MiB，但在总量中占比不大。最后，prompt 在分页池中确实只有一份：请求结束后，调度器块池中带哈希的整块数恰好等于输入长度除以 16（32、64、128），与 W 无关。

把实测块数与实测的后缀 slot 相加，W=128 时单个 item 的 KV 在输入 512、1024、2048 下分别为 98、154、266 MiB；如果每条 beam 都持有完整 KV，对应的计算值为 7,210、14,378、28,714 MiB（图 5-5 B）。

这部分显存的代价在于，即使没有请求也始终占用：后缀池和 scratch 随 W、S 线性增长，其他缓冲还略快一些。GR 路径可同时运行的 item 数由 B≤8 的准入契约限定（见 2.3），后缀池已按这一上限预分配，因此在准入范围内，它不会额外限制 item 数。

## 5.6 当前边界

这套缓存设计目前还有以下几处边界：

- 多级缓存只覆盖普通生成路径。持久 Beam 路径在启动时会拒绝 `kv_transfer_config`，因此 LMCache 的 CPU 内存缓存与跨实例 P2P 共享，目前只能用于普通 `generate()` 直通路径。让 Beam 路径接入外部 KV 存储，属于后续工作。
- 前缀复用以块为粒度。命中时至少要重算最后一块，不同 prompt 也只有在块边界之前完全一致时才能共享。
- 预分配不随负载伸缩。后缀池与 scratch 的大小在启动时由 `max_batch`、W 和 S 确定，没有请求时同样占用显存。
- 同一 batch 中的重复 prompt 需要额外处理。示例程序会为它们加上 `cache_salt`，避免在原子 Prefill 完成之前共享同一个 prompt 块。

## 5.7 小结

概括起来，vLLM-GR 的做法是：prompt KV 交给 vLLM，后缀 KV 由 GR 自己管理。前者依靠上游的分页池和 APC 完成跨请求复用，GR 不做改动；后者是一块地址固定的稠密池，配合写入、两级 attention 和两段式重排三类算子，GPU 与 NPU 遵守同一份布局契约。

在 OneRec-1.7B、W=128 的配置下，单个 item 的 KV 从完全不共享时的约 14 GiB 降至 154 MiB，GR 额外预分配的显存约为分页池的 2.3%；相同上下文再次提交时，端到端 p50 从 52.08 ms 降至 33.01 ms。

<!-- Markdown 预览采用首组相对链接；Jekyll 经 Liquid 处理后只保留带 baseurl 的第二组。 -->
<!-- {% unless site %} -->
[gr05-layout]: ../../assets/figures/vllm-gr/05-kv-memory-layout.svg
[gr05-two-level]: ../../assets/figures/vllm-gr/05-two-level-kv.svg
[gr05-reorder]: ../../assets/figures/vllm-gr/05-suffix-lifecycle-reorder.svg
[gr05-hit-miss]: ../../assets/figures/vllm-gr/05-prefix-hit-miss.svg
[gr05-sweep]: ../../assets/figures/vllm-gr/05-memory-sweep.svg
<!-- {% else %} -->
[gr05-layout]: <{{ '/assets/figures/vllm-gr/05-kv-memory-layout.svg' | relative_url }}>
[gr05-two-level]: <{{ '/assets/figures/vllm-gr/05-two-level-kv.svg' | relative_url }}>
[gr05-reorder]: <{{ '/assets/figures/vllm-gr/05-suffix-lifecycle-reorder.svg' | relative_url }}>
[gr05-hit-miss]: <{{ '/assets/figures/vllm-gr/05-prefix-hit-miss.svg' | relative_url }}>
[gr05-sweep]: <{{ '/assets/figures/vllm-gr/05-memory-sweep.svg' | relative_url }}>
<!-- {% endunless %} -->
