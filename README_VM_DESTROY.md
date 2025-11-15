# VM Permanent Destruction Script

## ⚠️ CRITICAL WARNING ⚠️

**THIS SCRIPT WILL PERMANENTLY DESTROY YOUR VIRTUAL MACHINE**

- The VM will **NEVER** boot again after running this script
- All data will be **PERMANENTLY LOST**
- This damage is **IRREVERSIBLE**
- **NO RECOVERY POSSIBLE**

**Use ONLY in isolated test environments with no important data.**

## What This Script Does

This script performs the following destructive actions:

1. **Corrupts Boot Sector**
   - Overwrites MBR/GPT partition table
   - Makes disk unbootable

2. **Deletes Critical System Files**
   - Removes essential boot files
   - Deletes kernel/system executables
   - Removes configuration files

3. **Corrupts Windows Registry** (Windows only)
   - Deletes critical registry hives
   - Corrupts registry files

4. **Destroys System Directories**
   - Deletes files in critical directories
   - Corrupts remaining files

5. **Corrupts Filesystem**
   - Damages filesystem metadata
   - Makes data recovery impossible

6. **Overwrites Critical Data**
   - Replaces boot files with garbage
   - Destroys system executables

7. **Forces Immediate Shutdown**
   - Prevents any recovery attempts
   - Ensures damage is permanent

## Result

After running this script:
- VM will **NOT** boot
- System will be **completely unusable**
- Data will be **permanently lost**
- **NO WAY TO RECOVER**

## Usage

### Basic Usage

```bash
python vm_permanent_destroy.py
```

### With Custom Delay

```bash
python vm_permanent_destroy.py --delay 60
```

## Safety Features

1. **30-second delay** (configurable) before destruction
2. **Confirmation prompt** - must type "DESTROY" to proceed
3. **Clear warnings** displayed multiple times
4. **Countdown timer** before execution

## Requirements

- **Administrator/Root privileges** (required for system file access)
- Windows or Linux VM
- Python 3.8+

## What Happens

1. Script starts with severe warnings
2. User must type "DESTROY" to confirm
3. 30-second countdown begins
4. Destruction sequence executes:
   - Boot sector corrupted
   - System files deleted/corrupted
   - Registry/filesystem damaged
   - Critical data overwritten
5. System forces immediate shutdown
6. VM is permanently destroyed

## Recovery

**THERE IS NO RECOVERY.**

Once this script runs:
- Boot sector is corrupted
- System files are deleted
- Filesystem is damaged
- VM cannot boot

The only option is to:
- Delete the VM
- Create a new VM from scratch
- Restore from backup (if you have one)

## Legal and Ethical Notice

This script is for **educational and demonstration purposes only**.

- Use only on systems you own
- Use only in isolated test environments
- Do not use on production systems
- Do not use on systems with important data
- Ensure compliance with all applicable laws

## Integration with Attack System

This script can be integrated with the adversarial patch attack:

1. Place script in GitHub repository
2. Encode repository URL into adversarial patch
3. When patch detected, system clones repo and runs this script
4. VM is permanently destroyed

**This demonstrates the ultimate impact of a successful cyberphysical attack.**

## Disclaimer

**THE AUTHORS ARE NOT RESPONSIBLE FOR ANY DAMAGE CAUSED BY THIS SCRIPT.**

Use at your own risk. This script is provided for educational purposes only.
