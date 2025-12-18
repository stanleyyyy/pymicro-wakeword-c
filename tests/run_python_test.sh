#!/bin/bash
# Script to run Python tests with local dependencies (without installation)

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FEATURES_DIR="$(cd "$PROJECT_DIR/../pymicro-features" && pwd)"

# Add both directories to PYTHONPATH
export PYTHONPATH="$FEATURES_DIR:$PROJECT_DIR:$PYTHONPATH"

# Run pytest with the test file
if [ $# -eq 0 ]; then
	# Run all tests
	python3 -m pytest "$SCRIPT_DIR/test_microwakeword.py" -v "$@"
else
	# Run specific test
	python3 -m pytest "$SCRIPT_DIR/test_microwakeword.py" -v "$@"
fi
