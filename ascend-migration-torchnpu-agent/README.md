# Ascend Migration torch_npu Agent

A Claude Code / OpenCode agent plugin for migrating deep learning models
from CPU/GPU to Huawei Ascend NPU via the `torch_npu` adaptation layer.

## Overview

This plugin automates the end-to-end migration workflow:

```
Analysis → CPU Baseline → NPU Migration → Verification → Report
```

It handles device mapping (cuda→npu), interface replacement, mixed precision
adaptation, distributed training conversion (nccl→hccl, DP→DDP),
environment setup (Docker + CANN + torch_npu), and verification reporting.

## Installation

### Claude Code

```bash
# Install from source
git clone <repo-url>
cd ascend-migration-torchnpu-agent
claude plugins install .
```

### OpenCode

```bash
# Copy to OpenCode skills directory
cp -r skills/* ~/.config/opencode/skills/
cp agents/ascend-migration-torchnpu-agent.md ~/.config/opencode/agents/
```

## Structure

```
ascend-migration-torchnpu-agent/
├── .claude-plugin/
│   └── plugin.json                    # Plugin metadata
├── agents/
│   └── ascend-migration-torchnpu-agent.md   # Subagent definition
├── skills/
│   ├── migration-ascend-torchnpu-skills/              # Main: workflow orchestration
│   │   └── SKILL.md
│   ├── migration-ascend-torchnpu-skills-migration-execution/  # Code migration
│   │   └── SKILL.md
│   ├── migration-ascend-torchnpu-skills-environment-setup/    # Environment setup
│   │   └── SKILL.md
│   ├── migration-ascend-torchnpu-skills-torch-npu-reference/  # API reference
│   │   └── SKILL.md
│   └── migration-ascend-torchnpu-skills-troubleshooting/       # Error diagnosis
│       └── SKILL.md
├── templates/
│   └── migration-report-template.md    # Report output template
├── hooks/
│   └── hooks.json                     # SessionStart + PostToolUse hooks
├── scripts/
│   ├── session-start.sh               # Injects NPU migration rules on session start
│   └── post-edit.sh                   # Validates CUDA→NPU changes after file edits
├── LICENSE                            # MIT
└── README.md                          # This file
```

## Usage

Trigger the agent with natural language:

- "migrate this model to run on Ascend NPU"
- "adapt my training script from CUDA to torch_npu"
- "set up an Ascend NPU environment for this project"
- "check if my model's interfaces are supported on NPU"

Or invoke directly:

- Claude Code: `/ascend-migration-torchnpu-agent`
- OpenCode: Press `Tab` to cycle to the agent

## Capabilities

| Area | Coverage |
|------|----------|
| **Device mapping** | cuda→npu, nccl→hccl, DataParallel→DDP, cudnn→delete |
| **Mixed precision** | torch.cuda.amp → torch.npu.amp |
| **Optimizers** | NpuFusedSGD, NpuFusedAdamW, NpuFusedAdam |
| **Environment** | Docker (CANN images), manual driver/CANN/torch_npu install |
| **Version matrix** | CANN 7.0~9.0 × PyTorch 2.0~2.10 × torch_npu |
| **Third-party libs** | transformers, accelerate, peft, trl (Ascend-native support) |
| **Model access** | ModelScope, HF-mirror, HuggingFace |
| **Verification** | Precision comparison tools, layer-by-layer validation |
| **Debugging** | Interface support query, error pattern recognition |

## Safety

- Never modifies third-party library source code
- Requires user confirmation for system-level operations
- Declares untested status when no NPU hardware is available
- Prioritizes mirrors (pip, ModelScope) for restricted network environments

## License

MIT — see [LICENSE](./LICENSE)
