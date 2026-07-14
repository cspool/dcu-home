# C100 three-request aggregate worst-result fixtures (superseded)

> **状态：AGGREGATE-ONLY / SUPERSEDED FOR INDIVIDUAL-REQUEST SCREEN。**
>
> 本卡只冻结 C100 `all 3` 的每档前三条聚合结果。C100 没有完成每档 50 条的
> 完整 `all`，因此本卡不能回答“完整测试集中每档最差单请求”。后续小样本
> 评估改用 H11.5+H10.8 完整 full-`all` 生成的
> [CURRENT_BEST_WORST_REQUEST_FIXTURES](../20260712_h11_5_h10_8_final_evidence/CURRENT_BEST_WORST_REQUEST_FIXTURES.md)。

## Identity and selection rule

This card freezes the worst archived aggregate result for each context bucket
from the current development performance best:

- wheel SHA256:
  `f877d08fdf2380a87298006c915d14077ca947225e50e5bcf56e028fc9075d80`;
- runtime: H10 profile enabled, S32 disabled, Hg3 disabled;
- evidence window: `all3_natural_600`, three rounds, three fixed records per
  bucket, 27/27 successful requests;
- selection within a bucket: minimum full-precision `output_throughput` among
  its three archived rounds.

`output_throughput` is a three-request aggregate emitted by the benchmark.  The
result file does not define an official per-record throughput, so this card does
not invent one by dividing a request's output length by a derived latency.  The
fixture unit is the unchanged first-three-record set for each JSONL dataset.

## Frozen per-bucket minima

| Bucket | Selected round/date | Output throughput | Duration | Input/output tokens | Input lengths | Output lengths | Result SHA256 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `4-8K` | `round-0001`, `20260713-053237` | `13.055002010433132 tok/s` | `13.175026695709676 s` | `19847 / 172` | `[7574,7548,4725]` | `[88,62,22]` | `868d0feb270dc38ba4642225e1246086ed62cce49d2a1172b93f181a59e49515` |
| `8-16K` | `round-0001`, `20260713-053401` | `15.879393477954984 tok/s` | `38.09969195863232 s` | `38677 / 605` | `[13962,14519,10196]` | `[95,12,498]` | `a283739c980376039cb863321379c726ab4ef3410f70cb756dc4dd400f7b1666` |
| `16-32K` | `round-0003`, `20260713-054234` | `10.027859684471808 tok/s` | `35.700539423618466 s` | `64238 / 358` | `[20574,22294,21370]` | `[23,259,76]` | `5bedcc15361d5bcfbc6b89c8b9cbbfafe27d01f677ce02dab229886de6a03d08` |

Result paths, relative to this evidence root:

- `all3_natural_600/round-0001/results/4-8K_throughput/result.json`
- `all3_natural_600/round-0001/results/8-16K_throughput/result.json`
- `all3_natural_600/round-0003/results/16-32K_throughput/result.json`

The `16-32K` entry is also the global minimum across all nine comparable C100
results.  The global maximum remains `15.890586153089119 tok/s` at
`round-0003/8-16K`, so the frozen current-best representative is
`T_mid=(15.890586153089119+10.027859684471808)/2 =
12.959222918780464 tok/s`.

## Dataset and first-three-record identity

`first-3 SHA256` hashes the exact raw bytes of the first three lines, including
their original line terminators.  Per-line hashes exclude the line terminator.

| Dataset | Full file SHA256 | First-3 raw bytes SHA256 | Line 1 / 2 / 3 SHA256 |
| --- | --- | --- | --- |
| `4-8K_throughput.jsonl` | `e53c4704ef89b99ae7f6855b14f92ab1607d532ada7a115fdf8158e022045c56` | `c0aee697d29470cdcffbb248e662d2d4ad99d2969c1e1e67eb68df90e054d37c` | `4a1de46e9a1574bfb89175ca688d3d225ee0fda658366a2233b4294146a14f1d` / `4e60dd244ca3658e8af0095452f271bce7d8ec4d142fbecdb59f43126d243afa` / `53efa30de5011e73a5f7a82ae1dc3e7bf8376564b7cce97488d8955c4b2e03e3` |
| `8-16K_throughput.jsonl` | `9513d8b6c14d7fea8aa769c031ee020288ba76b15c28d0090f903847acebf588` | `dc6ff1c3b5ccfc905df3f4db2716d78ff91b575d1607d3a6248167fe4f6edcde` | `cd9f390bae50678513e6f1bd5498b0f68f7f5fe4b432e1804390d3e7dcad0866` / `9c7f813e2bbb8c48f3d026ee5a5c36f4bedd5e0f8dd5291c4dad0586162c1195` / `3a98a545ddeda9905119c48b62688e9a2e54cc7ff897a7d3756d6ac417564ead` |
| `16-32K_throughput.jsonl` | `633ba4c8b4f500d2ab28094de42698c5494e5232f40eafcd119c0a314b44b936` | `c41c7ea76f6004be467905657d2839c39a77ecb4d7dd38b91603236c707ea50e` | `3c128207ee5a93bb84d2c804f840fe62c1f7d7a05434e058f4e258453a56f5bc` / `1594855fd7328b5ec046a78295996ff467187895b92eb3a8fc3441e834772952` / `93faccd8e565fbbedca575953dccc8457fe7ef8ed13c08d65f493d3deb2e33ac` |

## Use after supersession

Do not use these three-request aggregates as the current small-sample promotion
gate or as the worst request of a 50-record dataset.  They remain immutable
historical evidence for reproducing C100's old `all 3` window only.  Current
evaluation uses the three individual rows frozen in
`../20260712_h11_5_h10_8_final_evidence/CURRENT_BEST_WORST_REQUEST_FIXTURES.md`,
followed by a complete 50-request-per-bucket `all` comparison.
