# 文档导航

本目录按用途组织。当前实现以根目录 [README](../README.md) 及下方“架构”文档为准；`archive/` 仅用于追溯历史，不应作为当前控制语义或实验配置的依据。

## 我想理解当前系统

- [项目结构](architecture/project-structure.md)
- [Residual CEM-MPC cost function](architecture/cost-function.md)
- [Residual CEM-MPC 伪代码](architecture/mpc-pseudocode.md)
- [MPC 架构与默认配置演进](architecture/mpc-architecture-evolution.md)
- [Planner projection](architecture/planner-projection.md)

## 我想运行或复现实验

- [运行命令](guides/run-commands.md)
- [Model A 鲁棒性](guides/model-a-robustness.md)
- [Direct IK 鲁棒性](guides/direct-ik-robustness.md)
- [Model C 数据闭环](guides/model-c-workflow.md)
- [Model A replica 训练](guides/model-a-replica-training-5090.md)
- [UR5e 从采集到 CEM-MPC 全流程](guides/ur5e-end-to-end-workflow.md)

## 我想查论文实验或结果

- [论文实验总计划](experiments/paper-test-plan.md)
- [Delay-Aware MPC 论文实验操作手册](experiments/paper-delay-aware-experiments.md)
- [Delay-aware MPC 鲁棒性结果](experiments/delay-aware-robustness.md)
- [Cost 消融](experiments/cost-ablation.md) 与 [Threaded-ASAP Cost 消融](experiments/cost-ablation-threaded.md)
- [延迟、多速率与性能分析](experiments/latency-multirate-analysis.md)
- [旧 GRU 的四架构历史对比](experiments/four-architectures-legacy-gru.md)

## 安全与历史资料

- [不确定性软安全监督器](safety/uncertainty-soft-supervisor.md)
- [历史资料索引](archive/README.md)
