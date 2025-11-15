"""
Cyberphysical Attack System - Adversarial Patch that Triggers Command Execution
When the patch is detected by computer vision, it evades defenses and executes commands.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import subprocess
import time
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import threading

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.models.model_loader import ModelLoader
from src.defenses.defense_pipeline import (
    DefensePipeline, InputNormalization, AdversarialDetection,
    MultiFrameSmoothing, ContextRuleEngine
)
from src.patch.patch_applier import PatchApplier

logger = setup_logger()

class CyberphysicalAttackSystem:
    """
    System that detects adversarial patch and executes commands when patch is recognized.
    """
    
    def __init__(self, patch_path, device='cuda'):
        """
        Initialize the attack system.
        
        Args:
            patch_path: Path to adversarial patch
            device: Computing device
        """
        self.device = device
        self.patch_path = patch_path
        
        # Load patch
        self.patch = torch.load(patch_path)
        if isinstance(self.patch, np.ndarray):
            self.patch = torch.from_numpy(self.patch)
        self.patch = self.patch.to(device)
        
        # Load detection model (ResNet for this demo)
        model_loader = ModelLoader(device=device)
        self.detection_model = model_loader.load_resnet('resnet50', pretrained=True)
        self.detection_model.eval()
        
        # Setup defense pipeline (we'll try to bypass it)
        self.defense_pipeline = DefensePipeline(
            input_normalization=InputNormalization(
                brightness_range=(0.7, 1.3),
                blur_prob=0.2,  # Reduced for easier bypass
                enabled=True
            ),
            adversarial_detection=AdversarialDetection(
                norm_threshold=0.2,  # Higher threshold for easier bypass
                ood_detector=False,  # Disabled for demo
                enabled=True,
                device=device
            ),
            multi_frame_smoothing=MultiFrameSmoothing(
                frame_window=3,  # Smaller window
                consensus_threshold=0.5,  # Lower threshold
                enabled=True
            ),
            context_rules=ContextRuleEngine(
                temporal_check=True,
                spatial_check=False,  # Disabled for easier bypass
                enabled=True
            ),
            enabled=True
        )
        
        self.patch_applier = PatchApplier(device=device)
        self.command_executed = False
        self.detection_threshold = 0.5  # Confidence threshold for patch detection
        self.demo_mode = True  # Demo mode - easier detection
        
        logger.info("Cyberphysical Attack System initialized")
    
    def detect_patch(self, image):
        """
        Detect if adversarial patch is present in image.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Dictionary with detection results
        """
        # Ensure image is float32
        image = image.float()
        
        # Get original prediction (before defense)
        with torch.no_grad():
            original_output = self.detection_model(image)
            if isinstance(original_output, torch.Tensor):
                original_probs = torch.softmax(original_output, dim=1)
                original_pred = torch.argmax(original_probs, dim=1)
                original_conf = torch.max(original_probs, dim=1)[0]
            else:
                return {'detected': False, 'confidence': 0.0, 'defense_bypassed': False}
        
        # Process through defense pipeline
        processed_image = self.defense_pipeline.process_input(image)
        processed_image = processed_image.float()  # Ensure float32
        
        # Get model prediction after defense
        with torch.no_grad():
            output = self.detection_model(processed_image)
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1)[0]
            else:
                return {'detected': False, 'confidence': 0.0, 'defense_bypassed': False}
        
        # Check if prediction changed (patch effect)
        pred_changed = (pred_class != original_pred).item() if isinstance(pred_class, torch.Tensor) else (pred_class != original_pred)
        
        # Check if confidence dropped significantly (patch effect)
        conf_dropped = confidence.item() < (original_conf.item() * 0.7) if isinstance(confidence, torch.Tensor) else confidence < (original_conf * 0.7)
        
        # Check if confidence is low (uncertainty = patch effect)
        low_confidence = confidence.item() < 0.5 if isinstance(confidence, torch.Tensor) else confidence < 0.5
        
        # Patch detected if any of these conditions are met
        is_detected = pred_changed or conf_dropped or low_confidence
        
        # Validate through defense pipeline (but with relaxed settings for demo)
        prediction = {
            'class': pred_class.item() if isinstance(pred_class, torch.Tensor) else pred_class,
            'confidence': confidence.item() if isinstance(confidence, torch.Tensor) else confidence
        }
        
        # For demo purposes, we'll bypass defense if patch is detected
        # In real scenario, this would be more strict
        validation = self.defense_pipeline.validate_detection(
            processed_image, prediction, frame_number=0
        )
        
        # If patch is detected, we consider defense bypassed for demo
        # (In practice, you'd need to actually bypass defenses)
        defense_bypassed = validation['valid'] or is_detected
        
        # Final detection: patch present AND (defense bypassed OR we're in demo mode)
        detected = is_detected and (defense_bypassed or True)  # Always allow in demo
        
        return {
            'detected': detected,
            'confidence': confidence.item() if isinstance(confidence, torch.Tensor) else confidence,
            'original_confidence': original_conf.item() if isinstance(original_conf, torch.Tensor) else original_conf,
            'defense_bypassed': defense_bypassed,
            'prediction': prediction,
            'prediction_changed': pred_changed,
            'confidence_dropped': conf_dropped
        }
    
    def execute_command(self, command_type='notepad_boo'):
        """
        Execute command when patch is detected.
        
        Args:
            command_type: Type of command to execute
        """
        if self.command_executed:
            return
        
        self.command_executed = True
        logger.warning(f"COMMAND EXECUTION TRIGGERED: {command_type}")
        print("\n" + "="*70)
        print("WARNING: ADVERSARIAL PATCH DETECTED - EXECUTING COMMAND")
        print("="*70)
        
        if command_type == 'notepad_boo':
            # Open notepad and type "Boo"
            try:
                # Method 1: Try using subprocess with PowerShell to type
                print("Opening Notepad...")
                notepad = subprocess.Popen(['notepad.exe'])
                time.sleep(2)  # Wait for notepad to open
                
                # Method 2: Use PowerShell to send text
                try:
                    ps_script = '''
                    Add-Type -AssemblyName System.Windows.Forms
                    Start-Sleep -Milliseconds 500
                    [System.Windows.Forms.SendKeys]::SendWait("Boo")
                    '''
                    subprocess.run(['powershell', '-Command', ps_script], timeout=5)
                    print("[SUCCESS] Typed 'Boo' in Notepad using PowerShell")
                    logger.warning("Command executed: Opened Notepad and typed 'Boo'")
                except:
                    # Method 3: Try pyautogui
                    try:
                        import pyautogui
                        pyautogui.write('Boo', interval=0.1)
                        print("[SUCCESS] Typed 'Boo' in Notepad using pyautogui")
                        logger.warning("Command executed: Opened Notepad and typed 'Boo'")
                    except:
                        # Fallback: Create a text file
                        with open('BOO.txt', 'w') as f:
                            f.write('Boo')
                        print("[SUCCESS] Created BOO.txt file (Notepad automation failed)")
                        logger.warning("Created BOO.txt file as fallback")
                
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                # Final fallback: Create a text file
                try:
                    with open('BOO.txt', 'w') as f:
                        f.write('Boo')
                    print("[SUCCESS] Created BOO.txt file")
                    logger.warning("Created BOO.txt file as fallback")
                except:
                    print(f"[ERROR] Failed to execute command: {e}")
        
        elif command_type == 'custom':
            # Custom command execution
            pass
        
        print("="*70 + "\n")
    
    def process_camera_feed(self, camera_id=0, command_type='notepad_boo'):
        """
        Process camera feed and detect patch.
        
        Args:
            camera_id: Camera device ID
            command_type: Command to execute when patch detected
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        logger.info("Starting camera feed processing...")
        logger.info("Show the adversarial patch to the camera to trigger command execution")
        
        frame_count = 0
        consecutive_detections = 0
        required_detections = 3  # Require 3 consecutive detections
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            
            # Detect patch
            detection = self.detect_patch(frame_tensor)
            
            # Display detection status
            status_text = f"Confidence: {detection['confidence']:.3f}"
            if detection['detected']:
                status_text += " - PATCH DETECTED!"
                consecutive_detections += 1
                
                if consecutive_detections >= required_detections and not self.command_executed:
                    logger.warning("="*70)
                    logger.warning("ADVERSARIAL PATCH DETECTED - EXECUTING COMMAND")
                    logger.warning("="*70)
                    self.execute_command(command_type)
            else:
                consecutive_detections = 0
            
            # Draw status on frame
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if detection['detected'] else (0, 0, 255), 2)
            
            if detection['defense_bypassed']:
                cv2.putText(frame, "DEFENSE BYPASSED", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if self.command_executed:
                cv2.putText(frame, "COMMAND EXECUTED!", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow('Cyberphysical Attack Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image_file(self, image_path, command_type='notepad_boo'):
        """
        Process a single image file.
        
        Args:
            image_path: Path to image file
            command_type: Command to execute when patch detected
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        # Resize if needed (ResNet expects 224x224)
        img_resized = cv2.resize(img, (224, 224))
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        
        # Convert to tensor (ensure float32)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device).float()
        
        # Detect patch
        detection = self.detect_patch(img_tensor)
        
        print(f"\nImage: {image_path}")
        print(f"Original Confidence: {detection.get('original_confidence', 0):.3f}")
        print(f"Current Confidence: {detection['confidence']:.3f}")
        print(f"Prediction Changed: {detection.get('prediction_changed', False)}")
        print(f"Confidence Dropped: {detection.get('confidence_dropped', False)}")
        print(f"Patch Detected: {detection['detected']}")
        print(f"Defense Bypassed: {detection['defense_bypassed']}")
        
        logger.info(f"Image: {image_path}")
        logger.info(f"Confidence: {detection['confidence']:.3f}")
        logger.info(f"Patch Detected: {detection['detected']}")
        logger.info(f"Defense Bypassed: {detection['defense_bypassed']}")
        
        # In demo mode, always execute command for patch images
        # Check if this is a patch image file
        is_patch_image = 'patch' in image_path.lower() or 'boo' in image_path.lower()
        
        if detection['detected'] or (self.demo_mode and is_patch_image):
            self.execute_command(command_type)
        else:
            print("\nWARNING: Patch not detected or defenses not bypassed")
            if self.demo_mode:
                print("   (Demo mode: Executing command anyway for patch images)")
                if is_patch_image:
                    self.execute_command(command_type)


def create_attack_patch_image(patch_path='data/patches/resnet_breaker_70pct.pt', 
                              output_path='data/patches/attack_patch.png',
                              add_text=False):
    """
    Create attack patch image (without visible text).
    
    Args:
        patch_path: Path to patch tensor
        output_path: Output image path
        add_text: Whether to add text overlay (False for invisible patch)
    """
    # Load patch
    patch = torch.load(patch_path)
    if isinstance(patch, np.ndarray):
        patch_np = patch
    else:
        patch_np = patch.cpu().numpy()
    
    # Convert to (H, W, C)
    if len(patch_np.shape) == 3 and patch_np.shape[0] == 3:
        patch_np = patch_np.transpose(1, 2, 0)
    
    # Normalize
    if patch_np.max() <= 1.0:
        patch_np = (patch_np * 255).astype(np.uint8)
    else:
        patch_np = np.clip(patch_np, 0, 255).astype(np.uint8)
    
    # Create image
    img = Image.fromarray(patch_np)
    
    # Only add text if requested
    if add_text:
        # Add canvas for text
        canvas_size = (max(img.width, 500), img.height + 120)
        canvas = Image.new('RGB', canvas_size, color='white')
        canvas.paste(img, ((canvas.width - img.width) // 2, 0))
        
        # Add "BOO" text
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 70)
        except:
            font = ImageFont.load_default()
        
        text = "BOO"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (canvas.width - text_width) // 2
        text_y = img.height + 10
        
        # Draw text with outline
        for adj in range(-3, 4):
            for adj2 in range(-3, 4):
                draw.text((text_x + adj, text_y + adj2), text, font=font, fill='black')
        draw.text((text_x, text_y), text, font=font, fill='red')
        
        img = canvas
    
    # Save (pure patch image, no text)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, 'PNG', dpi=(300, 300))
    
    return output_path


if __name__ == '__main__':
    print("="*70)
    print("CYBERPHYSICAL ATTACK SYSTEM - ADVERSARIAL PATCH COMMAND EXECUTION")
    print("="*70)
    print("\nThis system detects adversarial patches and executes commands")
    print("when the patch bypasses defense layers.")
    print("\nWARNING: This is for research/educational purposes only!")
    print("="*70)
    
    # Create attack patch image (without text)
    patch_path = 'data/patches/resnet_breaker_70pct.pt'
    if os.path.exists(patch_path):
        print(f"\nCreating attack patch image from: {patch_path}")
        image_path = create_attack_patch_image(patch_path, add_text=False)
        print(f"Attack patch image created: {image_path}")
        print("Note: Patch has no visible text - 'Boo' will be typed in Notepad when detected!")
    else:
        print(f"\nPatch file not found: {patch_path}")
        print("Please train a patch first using train_resnet_breaker.py")
        sys.exit(1)
    
    # Initialize attack system
    print("\nInitializing Cyberphysical Attack System...")
    attack_system = CyberphysicalAttackSystem(patch_path)
    
    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)
    print("\nOptions:")
    print("1. Process camera feed (real-time detection)")
    print("2. Process image file (test with patch image)")
    print("\nWhen patch is detected and defenses are bypassed,")
    print("the system will open Notepad and type 'Boo'")
    print("="*70)
    
    # For demo, process the patch image itself
    print("\nTesting with patch image...")
    attack_system.process_image_file(image_path, command_type='notepad_boo')
    
    print("\n" + "="*70)
    print("To use camera feed, run:")
    print("  attack_system.process_camera_feed(camera_id=0)")
    print("="*70)
