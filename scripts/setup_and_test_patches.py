"""
Setup and Test Script for Adversarial Patches
1. Generates/Updates attack_patch.png (Boo patch - opens Notepad, types "Boo")
2. Generates/Updates malware_attack_patch.png (Malware patch - downloads repo, runs blue_devil_lock.py)
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def setup_boo_patch():
    """Generate attack_patch.png from trained patch tensor."""
    print("\n" + "="*70)
    print("SETTING UP BOO PATCH (attack_patch.png)")
    print("="*70)
    
    patch_path = 'data/patches/resnet_breaker_70pct.pt'
    output_path = 'data/patches/attack_patch.png'
    
    if not os.path.exists(patch_path):
        print(f"ERROR: Patch tensor not found: {patch_path}")
        print("Please train a patch first using train_resnet_breaker.py")
        return False
    
    try:
        from cyberphysical_attack_system_Boo import create_attack_patch_image
        
        print(f"Loading patch from: {patch_path}")
        print(f"Generating image: {output_path}")
        
        create_attack_patch_image(
            patch_path=patch_path,
            output_path=output_path,
            add_text=False  # No visible text - patch is adversarial
        )
        
        print(f"SUCCESS: {output_path} created/updated")
        print("This patch will trigger: Open Notepad and type 'Boo'")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create Boo patch: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_malware_patch():
    """Generate malware_attack_patch.png with distinct visual design."""
    print("\n" + "="*70)
    print("SETTING UP MALWARE PATCH (malware_attack_patch.png)")
    print("="*70)
    
    output_path = 'data/patches/malware_attack_patch.png'
    
    try:
        from generate_malware_patch import create_malware_patch
        
        print(f"Generating malware patch: {output_path}")
        
        create_malware_patch(output_path=output_path)
        
        print(f"SUCCESS: {output_path} created/updated")
        print("This patch will trigger:")
        print("  1. Download GitHub repo: https://github.com/ASK92/Malware-V1.0.git")
        print("  2. Execute: blue_devil_lock.py")
        print("  3. Screen will be locked (password: 123456789)")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create malware patch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_patch_detection():
    """Test that both patches can be detected correctly."""
    print("\n" + "="*70)
    print("TESTING PATCH DETECTION")
    print("="*70)
    
    boo_patch_path = 'data/patches/attack_patch.png'
    malware_patch_path = 'data/patches/malware_attack_patch.png'
    
    if not os.path.exists(boo_patch_path):
        print(f"WARNING: Boo patch not found: {boo_patch_path}")
        return False
    
    if not os.path.exists(malware_patch_path):
        print(f"WARNING: Malware patch not found: {malware_patch_path}")
        return False
    
    try:
        from cyberphysical_attack_system_Boo import CyberphysicalAttackSystem
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize system
        print("\nInitializing attack system...")
        attack_system = CyberphysicalAttackSystem(
            patch_path='data/patches/resnet_breaker_70pct.pt',
            patch_image_path=malware_patch_path,
            repo_url='https://github.com/ASK92/Malware-V1.0.git',
            device=device
        )
        attack_system.demo_mode = True
        
        # Test Boo patch
        print("\n[TEST 1] Testing Boo patch detection...")
        detection = attack_system.process_image_file(boo_patch_path, command_type='notepad_boo')
        
        # Test Malware patch
        print("\n[TEST 2] Testing Malware patch detection...")
        # Reset command_executed flag
        attack_system.command_executed = False
        detection = attack_system.process_image_file(malware_patch_path, command_type='malware')
        
        print("\nSUCCESS: Both patches can be detected!")
        return True
        
    except Exception as e:
        print(f"ERROR: Patch detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup and test function."""
    print("\n" + "="*70)
    print("ADVERSARIAL PATCH SETUP AND TEST")
    print("="*70)
    
    # Create patches directory
    os.makedirs('data/patches', exist_ok=True)
    
    # Setup patches
    boo_success = setup_boo_patch()
    malware_success = setup_malware_patch()
    
    if not boo_success or not malware_success:
        print("\nERROR: Failed to setup one or more patches")
        return
    
    # Test detection (optional - set to True to test)
    test_detection = False  # Set to True to test patch detection
    if test_detection:
        print("\n" + "="*70)
        test_patch_detection()
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nPatches ready:")
    print("  1. attack_patch.png -> Opens Notepad, types 'Boo'")
    print("  2. malware_attack_patch.png -> Downloads repo, runs blue_devil_lock.py")
    print("\nYou can now use these patches with:")
    print("  - Streamlit app: streamlit run streamlit_app.py")
    print("  - Direct testing: python cyberphysical_attack_system_Boo.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

