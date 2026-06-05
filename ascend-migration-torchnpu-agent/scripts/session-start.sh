#!/bin/bash
# SessionStart hook - injects NPU migration context reminders
# Runs at the beginning of every Claude Code session

cat << 'CONTEXT'
## ⚠️ Ascend NPU Migration Context

You have the `ascend-migration-torchnpu-agent` plugin active. When performing
model migration tasks, remember:

### Mandatory Principles
1. **Follow the 5-step workflow**: Analysis → CPU Baseline → NPU Migration → Verification → Report
2. **Never skip steps** - each step depends on the previous one
3. **Verify on CPU first** - never migrate to NPU before establishing CPU baseline
4. **Use official docs** - always check hiascend.com for latest version info
5. **Never fabricate data** - all results must come from actual execution
6. **Prioritize mirrors** - use ModelScope/HF-mirror, not HuggingFace directly

### Available Skills
Load these when context matches:
- `migration-ascend-torchnpu-skills` — overall workflow orchestration
- `migration-ascend-torchnpu-skills-migration-execution` — code changes & interface replacement
- `migration-ascend-torchnpu-skills-environment-setup` — Docker/CANN/torch_npu installation
- `migration-ascend-torchnpu-skills-torch-npu-reference` — API compatibility lookup

### Key Constraints
- NPU servers are typically ARM (aarch64) architecture
- Use pip mirrors: `-i https://mirrors.aliyun.com/pypi/simple/`
- cuda→npu, nccl→hccl, DataParallel→DDP, cudnn→delete
- NEVER modify third-party library source code
- When no NPU is available, state that code is untested
CONTEXT

echo "[ascend-migration] SessionStart context injected"
