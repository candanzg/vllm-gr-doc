# 6 竞品性能分析

```text
// 丁易
GPU 、 NPU
```

## 6.1 L20 上的 SGLang OneRec 离线基线

在单张 NVIDIA L20 上，使用 SGLang 0.5.22、OneRec-1.7B BF16、TP=1，完成 SID=3 的 288 组参数测试；每组测量 1000 个 batch。`beam` 为候选宽度，输入长度、batch 与其组合覆盖完整矩阵。测试通过离线 API 进行，以下时延是整个 batch 的同步完成时延，不是单请求延迟。

| 输入长度 | Beam | Batch | 平均 batch 时延 (ms) | P99 batch 时延 (ms) | 请求/秒 | 候选 token/秒 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 1 | 60.17 | 62.73 | 16.62 | 1595.40 |
| 1024 | 256 | 8 | 944.76 | 1676.97 | 8.47 | 6503.24 |
| 10240 | 1024 | 16 | 26102.78 | 27258.40 | 0.61 | 1883.02 |

完整矩阵见仓库中的 [L20 SGLang 数据](https://github.com/candanzg/vllm-gr-doc/blob/main/vLLM-GR%E4%BD%8E%E6%97%B6%E5%BB%B6%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E6%8E%A8%E7%90%86%E7%B3%BB%E7%BB%9F/performance_data/L20/sglang-onerec-sid3.csv)和[测试说明](https://github.com/candanzg/vllm-gr-doc/blob/main/vLLM-GR%E4%BD%8E%E6%97%B6%E5%BB%B6%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E6%8E%A8%E7%90%86%E7%B3%BB%E7%BB%9F/performance_data/L20/sglang-onerec-sid3.md)。由于站点配置排除 `performance_data`，数据链接指向 GitHub 仓库。

候选 token 吞吐包含所有 beam 候选；跨框架对比前需要确认输入、候选数和计时口径一致。本次未给出跨框架加速比。
