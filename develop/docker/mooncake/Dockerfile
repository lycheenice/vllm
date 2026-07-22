# 派生镜像:vLLM v0.25.0 + mooncake-transfer-engine(供 test2 / test4 使用)
# 基座已含 nixl;这里仅补 mooncake。★纯 CPU 构建★(不加 --gpus,pip 装 wheel 不需 GPU/设备)。
FROM docker.1ms.run/vllm/vllm-openai:v0.25.0

# mooncake 传输引擎(examples 检测的是 `import mooncake.engine`)。
# 只 pip 安装,不在 build 阶段做 import 验证(import 可能依赖运行时 RDMA 设备,
# 放到 docker run 阶段验证,保证 build 纯 CPU 且不因设备缺失而失败)。
RUN pip install --no-cache-dir mooncake-transfer-engine
