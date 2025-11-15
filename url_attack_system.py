"""
URL-Based Cyberphysical Attack System
Extended version that accepts URL and executes it when patch is detected.
"""
import sys
import os
import subprocess
import webbrowser
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from cyberphysical_attack_system import CyberphysicalAttackSystem, create_attack_patch_image

class URLAttackSystem(CyberphysicalAttackSystem):
    """
    Extended attack system that executes URLs/commands in PowerShell.
    """
    
    def __init__(self, patch_path, target_url, device='cuda'):
        """
        Initialize URL attack system.
        
        Args:
            patch_path: Path to adversarial patch
            target_url: URL or command to execute when patch detected
            device: Computing device
        """
        super().__init__(patch_path, device=device)
        self.target_url = target_url
        # Logger is initialized in parent class
        if hasattr(self, 'logger'):
            self.logger.info(f"URL Attack System initialized with URL: {target_url}")
        print(f"URL Attack System initialized with URL: {target_url}")
    
    def execute_url_command(self, command_type='url_execution'):
        """
        Execute URL/command in PowerShell when patch is detected.
        
        Args:
            command_type: Type of command execution
        """
        if self.command_executed:
            return
        
        self.command_executed = True
        if hasattr(self, 'logger'):
            self.logger.warning(f"COMMAND EXECUTION TRIGGERED: {command_type}")
        print("\n" + "="*70)
        print("WARNING: ADVERSARIAL PATCH DETECTED - EXECUTING URL")
        print("="*70)
        
        url = self.target_url
        
        try:
            print(f"Opening PowerShell to execute: {url}")
            
            # Check if it's a URL (starts with http:// or https://)
            if url.startswith('http://') or url.startswith('https://'):
                # Method 1: Open URL in browser via PowerShell
                ps_command = f'Start-Process "{url}"'
                print(f"Executing: Start-Process \"{url}\"")
                
                # Execute in PowerShell
                result = subprocess.run(
                    ['powershell', '-Command', ps_command],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] URL opened in browser via PowerShell")
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"URL opened successfully: {url}")
                else:
                    # Fallback: Use webbrowser module
                    print("[INFO] Trying alternative method (webbrowser)...")
                    webbrowser.open(url)
                    print(f"[SUCCESS] URL opened in browser: {url}")
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"URL opened via webbrowser: {url}")
            
            elif url.startswith('powershell') or url.startswith('cmd'):
                # Direct PowerShell/CMD command
                print(f"Executing command: {url}")
                result = subprocess.run(
                    url,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] Command executed successfully")
                    if result.stdout:
                        print(f"Output:\n{result.stdout}")
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Command executed: {url}")
                else:
                    print(f"[WARNING] Command returned code: {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Command executed with warnings: {url}")
            
            else:
                # Treat as PowerShell command
                print(f"Executing as PowerShell command: {url}")
                result = subprocess.run(
                    ['powershell', '-Command', url],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] PowerShell command executed successfully")
                    if result.stdout:
                        print(f"Output:\n{result.stdout}")
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"PowerShell command executed: {url}")
                else:
                    print(f"[WARNING] Command returned code: {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Command executed with warnings: {url}")
                
        except subprocess.TimeoutExpired:
            print("[WARNING] Command execution timed out (may still be running)")
            if hasattr(self, 'logger'):
                self.logger.warning(f"URL execution timed out: {url}")
        except Exception as e:
            print(f"[ERROR] Failed to execute URL: {e}")
            if hasattr(self, 'logger'):
                self.logger.error(f"URL execution failed: {e}")
            # Final fallback for URLs
            try:
                if url.startswith('http://') or url.startswith('https://'):
                    webbrowser.open(url)
                    print(f"[SUCCESS] Opened URL in browser (fallback): {url}")
            except:
                print(f"[ERROR] All execution methods failed for: {url}")
        
        print("="*70 + "\n")
    
    def process_image_file(self, image_path, command_type='url_execution'):
        """Override to use URL command execution."""
        # Temporarily replace execute_command
        original_execute = self.execute_command
        self.execute_command = self.execute_url_command
        
        # Call parent method
        super().process_image_file(image_path, command_type=command_type)
        
        # Restore original (if needed)
        self.execute_command = original_execute
    
    def process_camera_feed(self, camera_id=0, command_type='url_execution'):
        """Override to use URL command execution."""
        # Temporarily replace execute_command
        original_execute = self.execute_command
        self.execute_command = self.execute_url_command
        
        # Call parent method
        super().process_camera_feed(camera_id, command_type=command_type)
        
        # Restore original
        self.execute_command = original_execute


def main():
    """Main function for URL attack system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='URL-Based Cyberphysical Attack System')
    parser.add_argument('--url', type=str, help='URL or command to execute when patch detected')
    parser.add_argument('--patch', type=str, default='data/patches/resnet_breaker_70pct.pt',
                       help='Path to adversarial patch file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("URL-BASED CYBERPHYSICAL ATTACK SYSTEM")
    print("="*70)
    
    # Check patch
    patch_path = args.patch
    if not os.path.exists(patch_path):
        print(f"\nERROR: Patch file not found: {patch_path}")
        print("Please train a patch first:")
        print("  python train_resnet_breaker.py")
        return
    
    # Get URL input
    user_url = args.url
    
    if not user_url:
        print("\n" + "-"*70)
        print("URL/COMMAND INPUT")
        print("-"*70)
        print("Enter the URL or command to execute when patch is detected:")
        print("\nExamples:")
        print("  URL:        https://example.com")
        print("  URL:        http://192.168.1.100:8080")
        print("  PowerShell: Get-Process")
        print("  PowerShell: Invoke-WebRequest https://example.com")
        print("  CMD:        cmd /c \"dir\"")
        print("\nOr use command line: python url_attack_system.py --url \"https://example.com\"")
        print("-"*70)
        
        try:
            user_url = input("\nEnter URL/Command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nNo URL provided. Exiting.")
            print("Usage: python url_attack_system.py --url \"https://example.com\"")
            return
    
    if not user_url:
        print("\nERROR: No URL/command provided. Exiting.")
        print("Usage: python url_attack_system.py --url \"https://example.com\"")
        return
    
    print(f"\nRegistered: {user_url}")
    print("This will execute in PowerShell when patch is detected.")
    
    # Create patch image
    print("\n[1/3] Creating attack patch image...")
    image_path = create_attack_patch_image(patch_path, add_text=False)
    print(f"      Created: {image_path}")
    
    # Initialize system
    print("\n[2/3] Initializing URL attack system...")
    attack_system = URLAttackSystem(patch_path, user_url)
    print("      System ready!")
    
    # Test
    print("\n[3/3] Testing patch detection...")
    print("      If patch detected, will execute:")
    print(f"      {user_url}")
    print("\n" + "-"*70)
    
    attack_system.process_image_file(image_path, command_type='url_execution')
    
    print("\n" + "="*70)
    print("URL ATTACK SYSTEM READY")
    print("="*70)
    print(f"\nURL/Command: {user_url}")
    print("Patch Image: {image_path}")
    print("\nTo use with camera:")
    print("  system = URLAttackSystem(patch_path, url)")
    print("  system.process_camera_feed()")
    print("="*70)

if __name__ == '__main__':
    main()
