"""
Run the cyberphysical attack system - detects patch and executes commands.
"""
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from cyberphysical_attack_system import CyberphysicalAttackSystem, create_attack_patch_image

def main():
    print("="*70)
    print("CYBERPHYSICAL ATTACK SYSTEM")
    print("Adversarial Patch -> Command Execution")
    print("="*70)
    
    # Check if patch exists
    patch_path = 'data/patches/resnet_breaker_70pct.pt'
    if not os.path.exists(patch_path):
        print(f"\nERROR: Patch file not found: {patch_path}")
        print("Please train a patch first:")
        print("  python train_resnet_breaker.py")
        return
    
    # Create attack patch image (without visible text)
    print("\n[1/3] Creating attack patch image (invisible patch, no text)...")
    image_path = create_attack_patch_image(patch_path, add_text=False)
    print(f"      Created: {image_path}")
    print("      Note: Patch image has no visible text - command executes when detected!")
    
    # Initialize system
    print("\n[2/3] Initializing attack system...")
    attack_system = CyberphysicalAttackSystem(patch_path)
    print("      System ready!")
    
    # Test with image
    print("\n[3/3] Testing patch detection...")
    print("      Showing patch image to system...")
    print("      If patch is detected and defenses bypassed,")
    print("      Notepad will open and type 'Boo'")
    print("\n" + "-"*70)
    
    attack_system.process_image_file(image_path, command_type='notepad_boo')
    
    print("\n" + "="*70)
    print("ATTACK COMPLETE")
    print("="*70)
    print("\nTo use camera feed (real-time detection):")
    print("  python -c \"from cyberphysical_attack_system import CyberphysicalAttackSystem;")
    print("             system = CyberphysicalAttackSystem('data/patches/resnet_breaker_70pct.pt');")
    print("             system.process_camera_feed()\"")
    print("="*70)

if __name__ == '__main__':
    main()
