#!/bin/bash
# PostToolUse hook - validates after Write/Edit operations
# Runs after file writes or edits in Claude Code

FILE="$1"
if [ -z "$FILE" ]; then
  exit 0
fi

echo "[ascend-migration] PostToolUse: checking $FILE"

# Check for common CUDA references that should have been migrated
if grep -q '\.cuda()' "$FILE" 2>/dev/null; then
  echo "⚠️  WARNING: .cuda() calls still present in $FILE — may need .npu() replacement"
fi

if grep -q 'torch\.cuda\.' "$FILE" 2>/dev/null; then
  echo "⚠️  WARNING: torch.cuda.* calls still present in $FILE — may need torch.npu.* replacement"
fi

if grep -q 'backend.*=.*["'\"'"']nccl["'\"'"']' "$FILE" 2>/dev/null; then
  echo "⚠️  WARNING: nccl backend detected in $FILE — should be hccl for NPU"
fi

if grep -q 'DataParallel' "$FILE" 2>/dev/null; then
  echo "⚠️  WARNING: DataParallel detected in $FILE — NPU only supports DDP"
fi

if grep -q 'torch\.backends\.cudnn' "$FILE" 2>/dev/null; then
  echo "⚠️  WARNING: torch.backends.cudnn detected in $FILE — should be removed for NPU"
fi

echo "[ascend-migration] PostToolUse check complete"
