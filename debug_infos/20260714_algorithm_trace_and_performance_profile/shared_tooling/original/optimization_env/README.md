# 单卡吞吐优化实验环境

这个目录只做环境搭建和命令生成，不执行实验。它面向 `../start_vllm.sh` 和 `../run_throughput.sh`，用于后续按矩阵复现实验。

当前不包含 trace、patch trace 或 FX trace 初始化。

## 文件

- `env.example`：默认环境变量模板。
- `benchmark_matrix.tsv`：候选 `max_num_batched_tokens` 参数矩阵。
- `scripts/init_optimization_env.sh`：创建本目录需要的子目录，必要时生成 `.env`。
- `scripts/render_commands.sh`：按矩阵打印启动和 benchmark 命令；只打印，不执行。
- `scripts/check_no_speculative.py`：检查 benchmark 配置中没有 speculative 参数。
- `scripts/summarize_benchmark_results.py`：读取已有 `result.json` 并输出 Markdown 汇总；没有结果时只提示。
- `manifests/run_manifest_template.json`：后续记录一次实验的模板。

## 初始化

```bash
cd /data3/Projects/scnet_ssh/remote-home/testdata/optimization_env
./scripts/init_optimization_env.sh
```

该命令只创建目录、复制 `.env`、运行配置检查，不启动 vLLM。

## 生成命令

打印全部矩阵命令：

```bash
./scripts/render_commands.sh
```

只打印一个 case：

```bash
./scripts/render_commands.sh chunk_8192_all
```

输出的命令需要人工复制到 shell 中执行。脚本本身不会启动 server，也不会执行 `vllm bench serve`。

## 结果汇总

在已有结果落到 `results/<case_id>/<context>_throughput/result.json` 后，可以执行：

```bash
./scripts/summarize_benchmark_results.py
```

该脚本只读取已有文件，不启动实验。

