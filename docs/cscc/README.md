# CSCC 优化文档索引

当前保留四类有效材料：

1. [官方原版优化实施指南](OFFICIAL_BASE_OPTIMIZATION_GUIDE.md)：以
   `fa718036` 为唯一基线，说明逐文件修改、优先级、难度和验证方法。
2. [源码构建、启动与冷热缓存简明流程](BUILD_SERVE_CACHE_QUICKSTART.md)：统一给出
   新 wheel、导入路径、单卡/DP=2 启动和每版本冷编译后热重启的可用命令。
3. [600 行全量性能与精度报告](MODULAR_3K_PARITY.md)：记录 DCU 0 的最终
   150 条吞吐、110 条精度、3k 对比、规则审计和结果哈希。
4. [闭卷高收益速记](CLOSED_BOOK_HIGH_IMPACT_CHEATSHEET.md)：背诵时间不足时，
   按性能收益整理官方入口、关键 shape、最短实现思路、fallback 和修改技巧。

旧 `repro-minimal/499 行/qwen35_rocm_opt/三人闭卷` 文档已经删除，因为其代码
入口、性能数据和设备范围不再描述当前实现。构建、环境和第三方边界分别见根目录
[BUILD.md](../../BUILD.md)、[ENVIRONMENT.md](../../ENVIRONMENT.md) 和
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
