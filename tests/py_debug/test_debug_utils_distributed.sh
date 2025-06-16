#!/bin/bash
# Test script for distributed debugging utilities

# Set environment variables to simulate distributed environment
export RANK=0
export WORLD_SIZE=2

# Run tests for master process
echo "=== Testing debugger with master process (RANK=0) ==="
python -c "
from src.py_debug.debug_utils import setup_debugger, DistributedDebugger
import os

# Verify rank detection
assert DistributedDebugger.get_rank() == 0, 'Failed to detect rank'
assert DistributedDebugger.is_master() == True, 'Failed to detect master process'

# Verify debugger setup
debugger = setup_debugger(only_master=True)
print('Debugger setup successful for master process')
"

# Test with non-master process
export RANK=1
echo -e "\n=== Testing debugger with worker process (RANK=1) ==="
python -c "
from src.py_debug.debug_utils import setup_debugger, DistributedDebugger
import os

# Verify rank detection
assert DistributedDebugger.get_rank() == 1, 'Failed to detect rank'
assert DistributedDebugger.is_master() == False, 'Incorrectly detected as master process'

# Verify debugger setup - should get dummy debugger
debugger = setup_debugger(only_master=True)
from src.py_debug.debug_utils import DummyDebugger
assert isinstance(debugger, DummyDebugger), 'Did not get dummy debugger for worker'
print('Debugger setup correctly avoided real debugger for worker process')
"

# Clean up
unset RANK
unset WORLD_SIZE

echo -e "\n=== All tests completed ==="