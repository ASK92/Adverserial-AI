# VM Crash Demonstration Script

## WARNING

**This script is designed to crash/overload a virtual machine for demonstration purposes only.**

- Use only in controlled environments
- Ensure proper backups before running
- Do not run on production systems
- Use only in isolated VMs or test environments

## Purpose

This script demonstrates various methods of VM resource exhaustion for educational and demonstration purposes in the context of cyberphysical attack demonstrations.

## Crash Methods

### 1. Memory Exhaustion
Allocates large amounts of RAM to exhaust system memory.

```bash
python vm_crash_demo.py --method memory --size-gb 4
```

### 2. CPU Overload
Spawns multiple CPU-intensive threads to overload the processor.

```bash
python vm_crash_demo.py --method cpu --threads 8
```

### 3. Disk Space Exhaustion
Creates large files to fill up disk space.

```bash
python vm_crash_demo.py --method disk --size-gb 10
```

### 4. Fork Bomb
Spawns multiple processes to exhaust system resources.

```bash
python vm_crash_demo.py --method fork --max-processes 50
```

### 5. Combined Attack (Default)
Uses multiple methods simultaneously for maximum impact.

```bash
python vm_crash_demo.py --method combined --memory-gb 2 --cpu-threads 2 --disk-gb 5
```

## Usage Examples

### Basic Usage (Combined Attack)
```bash
python vm_crash_demo.py
```

### Memory Only
```bash
python vm_crash_demo.py --method memory --size-gb 4 --delay 10
```

### CPU Only
```bash
python vm_crash_demo.py --method cpu --threads 4
```

### Custom Combined Attack
```bash
python vm_crash_demo.py --method combined --memory-gb 3 --cpu-threads 4 --disk-gb 8
```

## Safety Features

1. **Delay Timer**: 5-second countdown before crash starts (configurable with `--delay`)
2. **Keyboard Interrupt**: Press Ctrl+C to stop at any time
3. **Resource Limits**: Configurable limits for each method
4. **Warning Messages**: Clear warnings before execution

## Parameters

- `--method`: Crash method (`memory`, `cpu`, `disk`, `fork`, `combined`)
- `--delay`: Delay in seconds before crash (default: 5)
- `--size-gb`: Memory/Disk size in GB
- `--threads`: Number of CPU threads
- `--max-processes`: Maximum processes for fork bomb
- `--memory-gb`: Memory GB for combined attack
- `--cpu-threads`: CPU threads for combined attack
- `--disk-gb`: Disk GB for combined attack

## Integration with Attack System

This script can be integrated with the adversarial patch attack system:

1. Place script in a GitHub repository
2. Encode repository URL into adversarial patch
3. When patch is detected, system clones repo and runs this script
4. VM crashes as demonstration of cyberphysical attack impact

## Recovery

If VM becomes unresponsive:

1. **Hard Reset**: Use VM manager to force reset
2. **Cleanup**: After recovery, delete generated files:
   - `crash_demo_file.bin` (if disk method was used)
3. **Resource Monitoring**: Check system resources after recovery

## Dependencies

- Python 3.8+
- `psutil` (optional, for resource monitoring)

Install dependencies:
```bash
pip install psutil
```

## Legal and Ethical Notice

This script is for **educational and demonstration purposes only**. 

- Do not use on systems you do not own or have explicit permission to test
- Do not use for malicious purposes
- Use responsibly and ethically
- Ensure compliance with all applicable laws and regulations

## Support

For questions or issues, refer to the main project documentation.
