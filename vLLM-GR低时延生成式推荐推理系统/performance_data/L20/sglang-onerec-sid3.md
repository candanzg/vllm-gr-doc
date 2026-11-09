# L20 / SGLang OneRec-1.7B 性能数据

`sglang-onerec-sid3.csv` 是单卡 NVIDIA L20 上 SGLang OneRec-1.7B 的 SID=3 离线测试结果。测试在 2026-10-29 至 2026-11-08（北京时间）进行。288 组参数均成功完成，每组测量 1000 个批次；历史上因隔离检查未通过而作废的尝试不计入结果。SID 4–8 不在本次数据范围内。

## 参数与环境

- SID：3 个输出 token；beam：32、64、128、256、512、1024；输入长度：128、256、512、1024、2048、4096、8192、10240 token；batch：1、2、4、6、8、16，合计 288 组。
- 单张 NVIDIA L20，tensor parallelism 1；OneRec-1.7B，模型和 KV cache 均为 BF16；SGLang 0.5.22，sglang-kernel 0.4.9，PyTorch 2.14.1，Transformers 5.19.0，CUDA runtime 13.0.96，驱动 580.159.03。
- 离线 API；prefix/radix cache 关闭；temperature=0、length_penalty=1、max_tokens=3、ignore_eos，随机种子 0，static memory fraction 0.6。输入使用 8 个预置批次循环。
- 每组预热至少 6 批、至多 30 批；按预热稳定性规则结束预热后测量 1000 批。批时延在 GPU 同步后计时，是整个 batch 的完成时间，不是单请求延迟。
- GPU 进程约每秒检查一次，握手测试还在完成时检查。该采样不能排除采样间极短的干扰；前 6 个配置由同一冻结环境下的早期运行导入，使用早期驱动，不含后续完成握手。具体来源见 CSV 的 `evidence_source`。

## CSV 字段及统计口径

`sid_tokens`、`beam`、`input_tokens`、`batch_size` 为测试形状；`measured_batches` 为测量批数。`batch_mean_ms`、`batch_p50_ms`、`batch_p90_ms`、`batch_p99_ms` 分别是 1000 个批次完成时延的均值及分位数，单位为毫秒。

`requests_per_second` = 全部测量批的请求总数 / 测量总秒数；`candidate_tokens_per_second` = 返回的全部候选 token 数 / 测量总秒数，包含 beam 候选。`total_output_tokens` 和 `total_measured_batch_ms` 可用于复算聚合候选吞吐。聚合吞吐不是各批吞吐的算术均值；候选 token 吞吐也不能直接作为单条最终输出的 token 吞吐。

## 可复核标识

- 输入 bundle SHA-256：`05ecec31cac1cc9dac27fb64a9d09fa789b57e7403a09af1857cceadc0543a66`。
- SGLang 0.5.22 wheel SHA-256：`3505e3744366201296f94d244b3e7d3b0bb91b69726dc15e994070adb338ea2f`；该安装不是 Git checkout，未记录可确认的 SGLang Git commit。
- 运行包快照 SHA-256：`da229ee1e56ea82f243f45625b1231311343b0eefd09e592adfe2eaea5b86314`。
- 握手驱动 SHA-256：`1664acc3718098c82ea28dd11b1fe1179950c5596533f6a1fe0b23ff95a9c577`；导入用例的早期驱动 SHA-256：`007847fdcf70dafca7bb75df68d472397f711ba95173d15cd8fdb9d80f95762c`。

这些数字用于描述本次 SGLang 离线批处理结果。其他框架或在线服务的数据必须先核对模型、输入、候选数和计时口径，才能比较；本次未得出跨框架加速比。
