# Running Python Tests Without Installation

## Prerequisites

1. **Build the pymicro-features C++ extension:**
   ```bash
   cd ../pymicro-features
   python3 setup.py build_ext --inplace
   ```
   This creates `micro_features_cpp.abi3.so` in the pymicro-features directory.

2. **Set PYTHONPATH to include both directories:**
   ```bash
   export PYTHONPATH="../pymicro-features:.:$PYTHONPATH"
   ```

## Running Tests

### Option 1: Using the helper script
```bash
cd /work/Stream800/!!!\ all\ hints\ and\ scripts\ !!!/tensorflow/pymicro-wakeword
./tests/run_python_test.sh
```

### Option 2: Manual command
```bash
cd /work/Stream800/!!!\ all\ hints\ and\ scripts\ !!!/tensorflow/pymicro-wakeword
PYTHONPATH="../pymicro-features:." python3 -m pytest tests/test_microwakeword.py -v
```

### Option 3: Run specific test
```bash
cd /work/Stream800/!!!\ all\ hints\ and\ scripts\ !!!/tensorflow/pymicro-wakeword
PYTHONPATH="../pymicro-features:." python3 -m pytest tests/test_microwakeword.py::test_process_streaming -v -k "okay_nabu"
```

### Option 4: Run debug script
```bash
cd /work/Stream800/!!!\ all\ hints\ and\ scripts\ !!!/tensorflow/pymicro-wakeword
PYTHONPATH="../pymicro-features:." python3 tests/debug_python.py
```

## How It Works

- `pymicro-wakeword` depends on `pymicro-features` (imported as `from pymicro_features import MicroFrontend`)
- `pymicro-features` has a C++ extension module (`micro_features_cpp`) that needs to be built
- By adding both directories to `PYTHONPATH`, Python can find:
  - `pymicro_wakeword` from the current directory
  - `pymicro_features` from `../pymicro-features`
  - The compiled `micro_features_cpp.abi3.so` extension

## Troubleshooting

If you get `ModuleNotFoundError: No module named 'micro_features_cpp'`:
- Make sure you've built the extension: `cd ../pymicro-features && python3 setup.py build_ext --inplace`
- Check that `micro_features_cpp.abi3.so` exists in the pymicro-features directory

If you get `ModuleNotFoundError: No module named 'pymicro_features'`:
- Make sure `PYTHONPATH` includes `../pymicro-features`
- Verify the path is correct relative to where you're running the command
