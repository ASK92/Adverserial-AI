"""
URL-Based Cyberphysical Attack System
Prompts for URL input, then executes it in PowerShell when patch is detected.
"""
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from cyberphysical_attack_system import CyberphysicalAttackSystem, create_attack_patch_image

def main():
    print("="*70)
    print("URL-BASED CYBERPHYSICAL ATTACK SYSTEM")
    print("Adversarial Patch -> URL Execution in PowerShell")
    print("="*70)
    
    # Check if patch exists
    patch_path = 'data/patches/resnet_breaker_70pct.pt'
    if not os.path.exists(patch_path):
        print(f"\nERROR: Patch file not found: {patch_path}")
        print("Please train a patch first:")
        print("  python train_resnet_breaker.py")
        return
    
    # Get URL input from user
    print("\n" + "-"*70)
    print("URL INPUT REQUIRED")
    print("-"*70)
    print("Enter the URL/command to execute when patch is detected:")
    print("Examples:")
    print("  - https://example.com")
    print("  - http://192.168.1.100:8080")
    print("  - powershell -Command \"Get-Process\"")
    print("  - cmd /c \"dir\"")
    print("-"*70)
    
    user_url = input("\nEnter URL/Command: ").strip()
    
    if not user_url:
        print("\nERROR: No URL provided. Exiting.")
        return
    
    print(f"\nURL/Command registered: {user_url}")
    print("This will be executed in PowerShell when patch is detected.")
    
    # Create attack patch image (without text)
    print("\n[1/3] Creating attack patch image (invisible patch, no text)...")
    image_path = create_attack_patch_image(patch_path, add_text=False)
    print(f"      Created: {image_path}")
    print("      Note: Patch image has no visible text - URL executes when detected!")
    
    # Initialize system with URL
    print("\n[2/3] Initializing attack system with URL...")
    attack_system = CyberphysicalAttackSystem(patch_path)
    attack_system.target_url = user_url  # Store the URL
    print("      System ready!")
    
    # Test with image
    print("\n[3/3] Testing patch detection...")
    print("      Showing patch image to system...")
    print("      If patch is detected and defenses bypassed,")
    print(f"      PowerShell will execute: {user_url}")
    print("\n" + "-"*70)
    
    # Override execute_command to use URL
    original_execute = attack_system.execute_command
    
    def execute_url_command(command_type='url_execution'):
        """Execute URL in PowerShell when patch is detected."""
        if attack_system.command_executed:
            return
        
        attack_system.command_executed = True
        attack_system.logger.warning(f"COMMAND EXECUTION TRIGGERED: {command_type}")
        print("\n" + "="*70)
        print("WARNING: ADVERSARIAL PATCH DETECTED - EXECUTING URL")
        print("="*70)
        
        url = attack_system.target_url
        
        try:
            print(f"Opening PowerShell to execute: {url}")
            
            # Check if it's a URL (starts with http:// or https://)
            if url.startswith('http://') or url.startswith('https://'):
                # Open URL in default browser via PowerShell
                ps_command = f'Start-Process "{url}"'
                print(f"Executing: Start-Process \"{url}\"")
            else:
                # Execute as PowerShell command
                ps_command = url
                print(f"Executing PowerShell command: {url}")
            
            # Execute in PowerShell
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command', ps_command],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("[SUCCESS] URL/Command executed successfully in PowerShell")
                if result.stdout:
                    print(f"Output: {result.stdout}")
                attack_system.logger.warning(f"URL executed successfully: {url}")
            else:
                print(f"[WARNING] Command executed with return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                # Still consider it executed
                attack_system.logger.warning(f"URL executed (with warnings): {url}")
                
        except subprocess.TimeoutExpired:
            print("[WARNING] Command execution timed out (may still be running)")
            attack_system.logger.warning(f"URL execution timed out: {url}")
        except Exception as e:
            print(f"[ERROR] Failed to execute URL: {e}")
            attack_system.logger.error(f"URL execution failed: {e}")
            # Try alternative method
            try:
                import webbrowser
                if url.startswith('http://') or url.startswith('https://'):
                    webbrowser.open(url)
                    print(f"[SUCCESS] Opened URL in browser: {url}")
            except:
                print(f"[ERROR] All execution methods failed for: {url}")
        
        print("="*70 + "\n")
    
    # Replace execute_command method
    attack_system.execute_command = execute_url_command
    
    # Process the patch image
    attack_system.process_image_file(image_path, command_type='url_execution')
    
    print("\n" + "="*70)
    print("URL ATTACK COMPLETE")
    print("="*70)
    print(f"\nURL/Command: {user_url}")
    print("Status: Executed when patch detected")
    print("\nTo use camera feed (real-time detection):")
    print("  python -c \"from url_attack_system import URLAttackSystem;")
    print(f"             system = URLAttackSystem('{patch_path}', '{user_url}');")
    print("             system.process_camera_feed()\"")
    print("="*70)

if __name__ == '__main__':
    main()

