#!/bin/bash

# Navigate to your project directory
cd /path/to/CS336_LM_From_Scratch/assignments/assignment1-basics

# Activate environment and run the test
echo "Starting BPE TinyStories test at $(date)"
uv run pytest tests/test_train_bpe.py::test_bpe_tinystories -v 2>&1 | tee test_output.log

# Capture exit code
TEST_EXIT_CODE=${PIPESTATUS[0]}

# Print completion message
echo "Test completed at $(date) with exit code: $TEST_EXIT_CODE"

# If test passed
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ Test PASSED"
else
    echo "✗ Test FAILED"
fi

# Auto-logout (kills the SSH session)
sleep 2
kill -HUP $PPID