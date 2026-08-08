# CSCC 优化文档索引

当前只保留三类有效材料：

1. [官方原版优化实施指南](OFFICIAL_BASE_OPTIMIZATION_GUIDE.md)：以
   `fa718036` 为唯一基线，说明逐文件修改、优先级、难度和验证方法。
2. [600 行全量性能与精度报告](MODULAR_3K_PARITY.md)：记录 DCU 0 的最终
   150 条吞吐、110 条精度、3k 对比、规则审计和结果哈希。
3. [可选 DP=2 配置](DP2_MULTI_REQUEST.md)：保留决赛多卡准备所需启动与
   benchmark 配置，明确历史数据和当前 R24 验证边界。

旧 `repro-minimal/499 行/qwen35_rocm_opt/三人闭卷` 文档已经删除，因为其代码
入口、性能数据和设备范围不再描述当前实现。构建、环境和第三方边界分别见根目录
[BUILD.md](../../BUILD.md)、[ENVIRONMENT.md](../../ENVIRONMENT.md) 和
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
