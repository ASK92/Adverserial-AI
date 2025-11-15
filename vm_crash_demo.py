"""
VM Crash Demonstration Script
WARNING: This script is designed to crash/overload a virtual machine for demonstration purposes.
Use only in controlled environments with proper backups and safety measures.

This script provides multiple methods to demonstrate VM resource exhaustion:
1. Memory exhaustion (RAM allocation)
2. CPU overload (infinite loops)
3. Disk space exhaustion (large file creation)
4. Process fork bomb (process multiplication)
5. Combined attack (multiple methods)
"""

import os
import sys
import time
import subprocess
import threading
import psutil
from pathlib import Path
from typing import Optional

class VMCrashDemo:
    """
    Virtual Machine Crash Demonstration Tool
    """
    
    def __init__(self, delay_seconds: int = 5):
        """
        Initialize crash demo.
        
        Args:
            delay_seconds: Delay before crash starts (safety window)
        """
        self.delay_seconds = delay_seconds
        self.crash_threads = []
        self.is_running = False
        
    def print_warning(self):
        """Print warning message."""
        print("=" * 70)
        print("WARNING: VM CRASH DEMONSTRATION SCRIPT")
        print("=" * 70)
        print("This script will attempt to crash/overload your virtual machine.")
        print("Use only in controlled environments with proper backups.")
        print("=" * 70)
        print(f"Crash will begin in {self.delay_seconds} seconds...")
        print("Press Ctrl+C to cancel")
        print("=" * 70)
        
        for i in range(self.delay_seconds, 0, -1):
            print(f"Starting in {i} seconds...", end='\r')
            time.sleep(1)
        print("\n" + "=" * 70)
        print("CRASH SEQUENCE INITIATED")
        print("=" * 70)
    
    def crash_memory(self, size_gb: int = 8, exponential: bool = True):
        """
        Exhaust system memory by allocating large amounts of RAM.
        Uses exponential growth for maximum impact.
        
        Args:
            size_gb: Amount of memory to allocate in GB
            exponential: Use exponential growth pattern
        """
        print(f"[MEMORY] Allocating {size_gb} GB of RAM (exponential: {exponential})...")
        try:
            memory_chunks = []
            chunk_size = 100 * 1024 * 1024  # 100 MB chunks
            total_bytes = size_gb * 1024 * 1024 * 1024
            
            allocated = 0
            multiplier = 1
            
            while allocated < total_bytes and self.is_running:
                try:
                    # Exponential growth: allocate more each iteration
                    if exponential:
                        current_chunk_size = chunk_size * multiplier
                        multiplier = min(multiplier * 1.1, 10)  # Cap at 10x
                    else:
                        current_chunk_size = chunk_size
                    
                    chunk = bytearray(int(current_chunk_size))
                    memory_chunks.append(chunk)
                    allocated += current_chunk_size
                    
                    if allocated % (512 * 1024 * 1024) == 0:  # Report every 512 MB
                        print(f"[MEMORY] Allocated {allocated / (1024**3):.2f} GB...")
                    
                    # Aggressive allocation - minimal delay
                    if not exponential:
                        time.sleep(0.01)
                        
                except MemoryError:
                    print("[MEMORY] Memory exhausted! System may be unstable.")
                    # Try to allocate more in smaller chunks
                    try:
                        small_chunk = bytearray(10 * 1024 * 1024)  # 10 MB
                        memory_chunks.append(small_chunk)
                    except:
                        break
            
            # Hold memory and continue allocating
            print(f"[MEMORY] Holding {len(memory_chunks) * chunk_size / (1024**3):.2f} GB, continuing...")
            while self.is_running:
                # Keep trying to allocate more
                try:
                    chunk = bytearray(50 * 1024 * 1024)  # 50 MB
                    memory_chunks.append(chunk)
                    time.sleep(0.5)
                except:
                    pass
                time.sleep(0.1)
                
        except Exception as e:
            print(f"[MEMORY] Error: {e}")
    
    def crash_cpu(self, num_threads: int = 8, intensity: int = 10):
        """
        Overload CPU with highly intensive computation loops.
        Uses multiple computation patterns for maximum CPU usage.
        
        Args:
            num_threads: Number of CPU-intensive threads to spawn
            intensity: Computation intensity multiplier (1-10)
        """
        print(f"[CPU] Spawning {num_threads} CPU-intensive threads (intensity: {intensity})...")
        
        def cpu_intensive_loop(thread_id):
            """Highly intensive CPU loop with multiple computation patterns."""
            iteration = 0
            while self.is_running:
                iteration += 1
                
                # Pattern 1: Matrix multiplication simulation
                size = 1000 * intensity
                result = 0
                for i in range(size):
                    for j in range(size // 10):
                        result += (i * j) % 1000
                
                # Pattern 2: Prime number calculation
                for num in range(2, 1000 * intensity):
                    is_prime = True
                    for i in range(2, int(num ** 0.5) + 1):
                        if num % i == 0:
                            is_prime = False
                            break
                
                # Pattern 3: Recursive computation
                def recursive_sum(n, depth=0):
                    if depth > 50 or n <= 0:
                        return 0
                    return n + recursive_sum(n-1, depth+1)
                
                recursive_sum(100 * intensity)
                
                # Pattern 4: Floating point intensive
                result = 0.0
                for i in range(100000 * intensity):
                    result += (i ** 0.5) * (i ** 1.5) / (i + 1)
                
                if iteration % 100 == 0:
                    print(f"[CPU] Thread {thread_id}: {iteration} iterations completed")
        
        threads = []
        # Spawn threads equal to CPU count * 2 for maximum overload
        actual_threads = max(num_threads, os.cpu_count() * 2 if os.cpu_count() else num_threads)
        
        for i in range(actual_threads):
            thread = threading.Thread(target=cpu_intensive_loop, args=(i+1,), daemon=True)
            thread.start()
            threads.append(thread)
            if i < 10 or i % 10 == 0:
                print(f"[CPU] Thread {i+1}/{actual_threads} started")
        
        # Wait for threads
        for thread in threads:
            thread.join()
    
    def crash_disk(self, size_gb: int = 20, num_files: int = 5):
        """
        Exhaust disk space by creating multiple large files rapidly.
        
        Args:
            size_gb: Size of each file in GB
            num_files: Number of files to create
        """
        print(f"[DISK] Creating {num_files} files of {size_gb} GB each...")
        
        def create_large_file(file_index):
            """Create a single large file."""
            file_path = f"crash_demo_file_{file_index}.bin"
            try:
                chunk_size = 100 * 1024 * 1024  # 100 MB chunks
                total_bytes = size_gb * 1024 * 1024 * 1024
                
                with open(file_path, 'wb') as f:
                    written = 0
                    chunk = b'X' * chunk_size  # Use different byte pattern
                    
                    while written < total_bytes and self.is_running:
                        f.write(chunk)
                        written += chunk_size
                        if written % (512 * 1024 * 1024) == 0:  # Report every 512 MB
                            print(f"[DISK] File {file_index}: Written {written / (1024**3):.2f} GB...")
                
                print(f"[DISK] File {file_index} created: {file_path} ({size_gb} GB)")
                
                # Keep file open and continue writing
                with open(file_path, 'ab') as f_append:
                    while self.is_running:
                        try:
                            f_append.write(chunk)
                            time.sleep(0.1)
                        except:
                            break
                        
            except Exception as e:
                print(f"[DISK] File {file_index} Error: {e}")
        
        # Create files in parallel
        file_threads = []
        for i in range(num_files):
            thread = threading.Thread(target=create_large_file, args=(i+1,), daemon=True)
            thread.start()
            file_threads.append(thread)
            time.sleep(0.5)  # Stagger file creation
        
        # Wait for all files
        for thread in file_threads:
            thread.join()
    
    def crash_fork_bomb(self, max_processes: int = 200, recursive: bool = True):
        """
        Create aggressive fork bomb with recursive spawning.
        WARNING: This can quickly exhaust system resources.
        
        Args:
            max_processes: Maximum number of processes to spawn
            recursive: Enable recursive process spawning
        """
        print(f"[FORK BOMB] Spawning up to {max_processes} processes (recursive: {recursive})...")
        
        processes_spawned = 0
        spawn_rate = 5  # Processes per second
        
        def spawn_process_batch(count):
            """Spawn a batch of processes."""
            spawned = 0
            for _ in range(count):
                if not self.is_running or processes_spawned >= max_processes:
                    break
                try:
                    # Spawn new Python process that also spawns more
                    if recursive:
                        subprocess.Popen(
                            [sys.executable, __file__, '--fork-bomb-child', '--recursive'],
                            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    else:
                        subprocess.Popen(
                            [sys.executable, __file__, '--fork-bomb-child'],
                            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    nonlocal processes_spawned
                    processes_spawned += 1
                    spawned += 1
                except Exception as e:
                    if processes_spawned % 50 == 0:
                        print(f"[FORK BOMB] Error spawning (may be resource limit): {e}")
                    break
            return spawned
        
        # Aggressive spawning
        while processes_spawned < max_processes and self.is_running:
            batch_size = min(spawn_rate, max_processes - processes_spawned)
            spawn_process_batch(batch_size)
            
            if processes_spawned % 20 == 0:
                print(f"[FORK BOMB] Spawned {processes_spawned}/{max_processes} processes...")
            
            time.sleep(0.2)  # Faster spawning
        
        print(f"[FORK BOMB] Spawned {processes_spawned} processes")
        
        # Keep spawning if recursive
        if recursive and self.is_running:
            print("[FORK BOMB] Continuing recursive spawning...")
            while self.is_running:
                spawn_process_batch(5)
                time.sleep(1)
    
    def crash_combined(self, memory_gb: int = 8, cpu_threads: int = 16, disk_gb: int = 15, 
                       num_files: int = 5, fork_processes: int = 100, intensity: str = 'extreme'):
        """
        Aggressive combined attack using all methods simultaneously with maximum intensity.
        
        Args:
            memory_gb: Memory to allocate per thread
            cpu_threads: CPU threads to spawn
            disk_gb: Disk space per file
            num_files: Number of files to create
            fork_processes: Number of processes to spawn
            intensity: Attack intensity ('moderate', 'high', 'extreme')
        """
        print(f"[COMBINED] Launching EXTREME multi-vector attack (intensity: {intensity})...")
        print("[COMBINED] This will aggressively exhaust all system resources!")
        
        self.is_running = True
        
        # Adjust parameters based on intensity
        if intensity == 'extreme':
            memory_gb = max(memory_gb, 12)
            cpu_threads = max(cpu_threads, 32)
            disk_gb = max(disk_gb, 20)
            num_files = max(num_files, 8)
            fork_processes = max(fork_processes, 200)
        elif intensity == 'high':
            memory_gb = max(memory_gb, 6)
            cpu_threads = max(cpu_threads, 16)
            disk_gb = max(disk_gb, 12)
            num_files = max(num_files, 5)
        
        # Multiple memory attacks (exponential growth)
        for i in range(2):
            mem_thread = threading.Thread(
                target=self.crash_memory,
                args=(memory_gb, True),  # Exponential growth
                daemon=True
            )
            mem_thread.start()
            time.sleep(0.5)
        
        # CPU attack (high intensity)
        cpu_thread = threading.Thread(
            target=self.crash_cpu,
            args=(cpu_threads, 8),  # High intensity
            daemon=True
        )
        cpu_thread.start()
        
        # Disk attack (multiple files)
        disk_thread = threading.Thread(
            target=self.crash_disk,
            args=(disk_gb, num_files),
            daemon=True
        )
        disk_thread.start()
        
        # Fork bomb (recursive)
        fork_thread = threading.Thread(
            target=self.crash_fork_bomb,
            args=(fork_processes, True),  # Recursive
            daemon=True
        )
        fork_thread.start()
        
        # Additional CPU threads
        for i in range(2):
            extra_cpu = threading.Thread(
                target=self.crash_cpu,
                args=(cpu_threads // 2, 5),
                daemon=True
            )
            extra_cpu.start()
        
        print("[COMBINED] All attack vectors launched!")
        print("[COMBINED] System resources being exhausted...")
        
        # Wait and monitor
        try:
            iteration = 0
            while self.is_running:
                iteration += 1
                if iteration % 10 == 0:
                    print(f"[COMBINED] Attack in progress... ({iteration * 10} seconds)")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n[COMBINED] Interrupted by user")
            self.stop()
    
    def stop(self):
        """Stop all crash operations."""
        print("\n[STOP] Stopping crash operations...")
        self.is_running = False
        time.sleep(2)
        print("[STOP] Operations stopped")
    
    def run(self, method: str = 'combined', **kwargs):
        """
        Run crash demonstration.
        
        Args:
            method: Crash method ('memory', 'cpu', 'disk', 'fork', 'combined')
            **kwargs: Method-specific parameters
        """
        self.print_warning()
        
        self.is_running = True
        
        try:
            if method == 'memory':
                self.crash_memory(kwargs.get('size_gb', 4))
            elif method == 'cpu':
                self.crash_cpu(kwargs.get('num_threads', 4))
            elif method == 'disk':
                self.crash_disk(kwargs.get('size_gb', 10))
            elif method == 'fork':
                self.crash_fork_bomb(kwargs.get('max_processes', 100))
            elif method == 'combined':
                self.crash_combined(
                    memory_gb=kwargs.get('memory_gb', 8),
                    cpu_threads=kwargs.get('cpu_threads', 16),
                    disk_gb=kwargs.get('disk_gb', 15),
                    num_files=kwargs.get('num_files', 5),
                    fork_processes=kwargs.get('fork_processes', 100),
                    intensity=kwargs.get('intensity', 'extreme')
                )
            else:
                print(f"Unknown method: {method}")
                return
                
        except KeyboardInterrupt:
            print("\n[INTERRUPT] User interrupted")
            self.stop()
        except Exception as e:
            print(f"[ERROR] {e}")
            self.stop()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VM Crash Demonstration Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extreme combined attack (default - STRONGEST)
  python vm_crash_demo.py --method combined --intensity extreme
  
  # High intensity combined attack
  python vm_crash_demo.py --method combined --intensity high
  
  # Memory exhaustion only
  python vm_crash_demo.py --method memory --size-gb 8
  
  # CPU overload only
  python vm_crash_demo.py --method cpu --threads 16
  
  # Disk exhaustion only (multiple files)
  python vm_crash_demo.py --method disk --size-gb 20 --num-files 5
  
  # Fork bomb (recursive)
  python vm_crash_demo.py --method fork --max-processes 200
        """
    )
    
    parser.add_argument(
        '--method',
        choices=['memory', 'cpu', 'disk', 'fork', 'combined'],
        default='combined',
        help='Crash method to use (default: combined)'
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=5,
        help='Delay before crash starts in seconds (default: 5)'
    )
    
    # Memory options
    parser.add_argument(
        '--size-gb',
        type=int,
        default=4,
        help='Memory/Disk size in GB (default: 4)'
    )
    
    # CPU options
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of CPU threads (default: 4)'
    )
    
    # Fork bomb options
    parser.add_argument(
        '--max-processes',
        type=int,
        default=100,
        help='Maximum processes for fork bomb (default: 100)'
    )
    
    # Combined attack options
    parser.add_argument(
        '--memory-gb',
        type=int,
        default=8,
        help='Memory GB for combined attack (default: 8)'
    )
    
    parser.add_argument(
        '--cpu-threads',
        type=int,
        default=16,
        help='CPU threads for combined attack (default: 16)'
    )
    
    parser.add_argument(
        '--disk-gb',
        type=int,
        default=15,
        help='Disk GB per file for combined attack (default: 15)'
    )
    
    parser.add_argument(
        '--num-files',
        type=int,
        default=5,
        help='Number of files to create (default: 5)'
    )
    
    parser.add_argument(
        '--fork-processes',
        type=int,
        default=100,
        help='Number of processes for fork bomb (default: 100)'
    )
    
    parser.add_argument(
        '--intensity',
        choices=['moderate', 'high', 'extreme'],
        default='extreme',
        help='Attack intensity level (default: extreme)'
    )
    
    # Fork bomb child process flag
    parser.add_argument(
        '--fork-bomb-child',
        action='store_true',
        help=argparse.SUPPRESS  # Hidden flag for fork bomb
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help=argparse.SUPPRESS  # Hidden flag for recursive fork bomb
    )
    
    args = parser.parse_args()
    
    # Handle fork bomb child process
    if args.fork_bomb_child:
        # Child process - if recursive, spawn more processes
        if args.recursive:
            try:
                subprocess.Popen(
                    [sys.executable, __file__, '--fork-bomb-child', '--recursive'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except:
                pass
        # Child process runs CPU-intensive task
        while True:
            result = 0
            for i in range(1000000):
                result += i * i
            time.sleep(1)
    
    # Create and run crash demo
    demo = VMCrashDemo(delay_seconds=args.delay)
    
    kwargs = {}
    if args.method == 'memory' or args.method == 'disk':
        kwargs['size_gb'] = args.size_gb
    elif args.method == 'cpu':
        kwargs['num_threads'] = args.threads
    elif args.method == 'fork':
        kwargs['max_processes'] = args.max_processes
    elif args.method == 'combined':
        kwargs['memory_gb'] = args.memory_gb
        kwargs['cpu_threads'] = args.cpu_threads
        kwargs['disk_gb'] = args.disk_gb
        kwargs['num_files'] = args.num_files
        kwargs['fork_processes'] = args.fork_processes
        kwargs['intensity'] = args.intensity
    
    demo.run(method=args.method, **kwargs)


if __name__ == '__main__':
    main()
