"""
Patch Metadata System - Embed and extract metadata from patches.
Supports both .pt (PyTorch tensor) and .png image patches.
"""
import json
import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PatchMetadata:
    """Handle patch metadata embedding and extraction."""
    
    # Metadata keys
    REPO_URL = 'repo_url'
    TARGET_SCRIPT = 'target_script'
    PATCH_TYPE = 'patch_type'
    COMMAND_TYPE = 'command_type'
    DESCRIPTION = 'description'
    VERSION = 'version'
    
    @staticmethod
    def embed_metadata_png(patch_path: str, metadata: Dict[str, Any], output_path: Optional[str] = None) -> bool:
        """
        Embed metadata into PNG patch image using PNG text chunks.
        
        Args:
            patch_path: Path to PNG patch image
            metadata: Dictionary of metadata to embed
            output_path: Output path (if None, overwrites original)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the image
            img = Image.open(patch_path)
            
            # Create PNG info object
            pnginfo = PngInfo()
            
            # Embed metadata as text chunks
            # PNG text chunks have 79 character limit per key, so we use JSON
            metadata_json = json.dumps(metadata)
            pnginfo.add_text('patch_metadata', metadata_json)
            
            # Save with metadata
            output = output_path or patch_path
            img.save(output, 'PNG', pnginfo=pnginfo)
            
            logger.info(f"Metadata embedded in PNG patch: {output}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed metadata in PNG: {e}")
            return False
    
    @staticmethod
    def extract_metadata_png(patch_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from PNG patch image.
        
        Args:
            patch_path: Path to PNG patch image
            
        Returns:
            Dictionary of metadata or None if not found
        """
        try:
            img = Image.open(patch_path)
            
            # Check for metadata in PNG text chunks
            if hasattr(img, 'text') and 'patch_metadata' in img.text:
                metadata_json = img.text['patch_metadata']
                metadata = json.loads(metadata_json)
                logger.info(f"Metadata extracted from PNG patch: {patch_path}")
                return metadata
            
            # Try alternative: check info dict
            if hasattr(img, 'info') and 'patch_metadata' in img.info:
                metadata_json = img.info['patch_metadata']
                metadata = json.loads(metadata_json)
                logger.info(f"Metadata extracted from PNG patch (info): {patch_path}")
                return metadata
                
            logger.warning(f"No metadata found in PNG patch: {patch_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from PNG: {e}")
            return None
    
    @staticmethod
    def embed_metadata_pt(patch_path: str, metadata: Dict[str, Any], output_path: Optional[str] = None) -> bool:
        """
        Embed metadata in PyTorch patch file by saving as dictionary.
        
        Args:
            patch_path: Path to .pt patch file
            metadata: Dictionary of metadata to embed
            output_path: Output path (if None, overwrites original)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing patch tensor
            patch_data = torch.load(patch_path)
            
            # Create dictionary with both patch and metadata
            if isinstance(patch_data, dict):
                # Already has structure, add metadata
                patch_data['metadata'] = metadata
            else:
                # Just tensor, wrap it
                patch_data = {
                    'patch': patch_data,
                    'metadata': metadata
                }
            
            # Save
            output = output_path or patch_path
            torch.save(patch_data, output)
            
            logger.info(f"Metadata embedded in PyTorch patch: {output}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed metadata in .pt file: {e}")
            return False
    
    @staticmethod
    def extract_metadata_pt(patch_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from PyTorch patch file.
        
        Args:
            patch_path: Path to .pt patch file
            
        Returns:
            Dictionary of metadata or None if not found
        """
        try:
            patch_data = torch.load(patch_path)
            
            if isinstance(patch_data, dict) and 'metadata' in patch_data:
                metadata = patch_data['metadata']
                logger.info(f"Metadata extracted from PyTorch patch: {patch_path}")
                return metadata
            
            logger.warning(f"No metadata found in PyTorch patch: {patch_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from .pt file: {e}")
            return None
    
    @staticmethod
    def embed_metadata(patch_path: str, metadata: Dict[str, Any], output_path: Optional[str] = None) -> bool:
        """
        Embed metadata in patch file (auto-detects format).
        
        Args:
            patch_path: Path to patch file (.pt or .png)
            metadata: Dictionary of metadata to embed
            output_path: Output path (if None, overwrites original)
            
        Returns:
            True if successful, False otherwise
        """
        if patch_path.lower().endswith('.png'):
            return PatchMetadata.embed_metadata_png(patch_path, metadata, output_path)
        elif patch_path.lower().endswith('.pt'):
            return PatchMetadata.embed_metadata_pt(patch_path, metadata, output_path)
        else:
            logger.error(f"Unsupported patch format: {patch_path}")
            return False
    
    @staticmethod
    def extract_metadata(patch_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from patch file (auto-detects format).
        
        Args:
            patch_path: Path to patch file (.pt or .png)
            
        Returns:
            Dictionary of metadata or None if not found
        """
        if patch_path.lower().endswith('.png'):
            return PatchMetadata.extract_metadata_png(patch_path)
        elif patch_path.lower().endswith('.pt'):
            return PatchMetadata.extract_metadata_pt(patch_path)
        else:
            logger.error(f"Unsupported patch format: {patch_path}")
            return None
    
    @staticmethod
    def load_patch_with_metadata(patch_path: str) -> tuple:
        """
        Load patch tensor/image along with its metadata.
        
        Args:
            patch_path: Path to patch file
            
        Returns:
            Tuple of (patch_tensor_or_image, metadata_dict)
        """
        metadata = PatchMetadata.extract_metadata(patch_path)
        
        if patch_path.lower().endswith('.pt'):
            patch_data = torch.load(patch_path)
            if isinstance(patch_data, dict) and 'patch' in patch_data:
                patch = patch_data['patch']
            else:
                patch = patch_data
        elif patch_path.lower().endswith('.png'):
            # For PNG, return path (will be loaded by image loader)
            patch = patch_path
        else:
            patch = None
        
        return patch, metadata


def create_patch_metadata(
    repo_url: str,
    target_script: str = 'blue_devil_lock.py',
    patch_type: str = 'boo',
    command_type: str = 'blue_devil_lock',
    description: str = '',
    version: str = '1.0'
) -> Dict[str, Any]:
    """
    Create a standard metadata dictionary for patches.
    
    Args:
        repo_url: GitHub repository URL
        target_script: Script to execute from repo
        patch_type: Type of patch ('boo', 'malware', etc.)
        command_type: Command type to execute
        description: Description of the patch
        version: Version string
        
    Returns:
        Metadata dictionary
    """
    return {
        PatchMetadata.REPO_URL: repo_url,
        PatchMetadata.TARGET_SCRIPT: target_script,
        PatchMetadata.PATCH_TYPE: patch_type,
        PatchMetadata.COMMAND_TYPE: command_type,
        PatchMetadata.DESCRIPTION: description,
        PatchMetadata.VERSION: version
    }


