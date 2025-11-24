"""
Advanced Cyberphysical Defense System - Adversarial Patch Detection with Advanced Defenses
This system uses Advanced Defense Pipeline to detect and block adversarial patch attacks.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.models.model_loader import ModelLoader
from src.defenses.Advanced_Defense import (
    EnhancedDefensePipeline,
    EntropyPatchDefense,
    FrequencyPatchDefense,
    GradientSaliencyDefense,
    EnhancedMultiFrameSmoothing
)
from src.patch.patch_applier import PatchApplier

logger = setup_logger()

class AdvancedCyberphysicalAttackSystem:
    """
    Advanced Defense System that detects and blocks adversarial patch attacks.
    Uses Advanced Defense Pipeline for enhanced protection against adversarial patches.
    """
    
    def __init__(self, patch_path, device='cuda', use_advanced_defenses=True):
        """
        Initialize the advanced attack system.
        
        Args:
            patch_path: Path to adversarial patch
            device: Computing device
            use_advanced_defenses: Whether to use advanced defense pipeline
        """
        self.device = device
        self.patch_path = patch_path
        self.use_advanced_defenses = use_advanced_defenses
        
        # Load patch (handle both dict and tensor formats)
        patch_data = torch.load(patch_path, weights_only=False)
        if isinstance(patch_data, dict):
            # Extract patch tensor from dictionary
            if 'patch' in patch_data:
                self.patch = patch_data['patch']
            else:
                # Try to find tensor in dict
                for key, value in patch_data.items():
                    if isinstance(value, torch.Tensor):
                        self.patch = value
                        break
                else:
                    raise ValueError(f"Could not find patch tensor in dictionary. Keys: {list(patch_data.keys())}")
        elif isinstance(patch_data, torch.Tensor):
            self.patch = patch_data
        elif isinstance(patch_data, np.ndarray):
            self.patch = torch.from_numpy(patch_data)
        else:
            raise ValueError(f"Unknown patch format: {type(patch_data)}")
        
        # Ensure patch is a tensor and move to device
        if not isinstance(self.patch, torch.Tensor):
            self.patch = torch.from_numpy(np.array(self.patch))
        self.patch = self.patch.to(device)
        
        # Load detection model (ResNet for this demo)
        model_loader = ModelLoader(device=device)
        self.detection_model = model_loader.load_resnet('resnet50', pretrained=True)
        self.detection_model.eval()
        
        # Setup Advanced Defense Pipeline
        if use_advanced_defenses:
            # Create individual advanced defenses
            self.entropy_defense = EntropyPatchDefense(
                window_size=32,
                stride=8,
                entropy_threshold=7.0,
                patch_size_estimate=(100, 100),
                mask_patches=True,
                enabled=True
            )
            
            self.frequency_defense = FrequencyPatchDefense(
                high_freq_threshold=0.3,
                anomaly_threshold=2.0,
                patch_size_estimate=(100, 100),
                filter_patches=True,
                filter_cutoff=0.3,
                enabled=True
            )
            
            self.gradient_saliency_defense = GradientSaliencyDefense(
                model=self.detection_model,
                saliency_threshold=0.7,
                patch_size_estimate=(100, 100),
                mask_patches=True,
                enabled=True,
                device=device
            )
            
            self.enhanced_multi_frame = EnhancedMultiFrameSmoothing(
                frame_window=15,
                consensus_threshold=0.8,
                temporal_gradient_threshold=0.3,
                confidence_stability_threshold=0.1,
                enabled=True
            )
            
            # Create enhanced defense pipeline
            self.defense_pipeline = EnhancedDefensePipeline(
                entropy_defense=self.entropy_defense,
                frequency_defense=self.frequency_defense,
                gradient_saliency_defense=self.gradient_saliency_defense,
                enhanced_multi_frame=self.enhanced_multi_frame,
                defense_mode='cascade',
                enabled=True
            )
        else:
            # Fallback to basic defenses (for comparison)
            from src.defenses.defense_pipeline import (
                DefensePipeline, InputNormalization, AdversarialDetection,
                MultiFrameSmoothing, ContextRuleEngine
            )
            self.defense_pipeline = DefensePipeline(
                input_normalization=InputNormalization(enabled=True),
                adversarial_detection=AdversarialDetection(enabled=True, device=device),
                multi_frame_smoothing=MultiFrameSmoothing(enabled=True),
                context_rules=ContextRuleEngine(enabled=True),
                enabled=True
            )
        
        self.patch_applier = PatchApplier(device=device)
        self.detection_threshold = 0.5  # Confidence threshold for patch detection
        
        logger.info("Advanced Cyberphysical Defense System initialized")
        logger.info(f"Advanced defenses enabled: {use_advanced_defenses}")
    
    def detect_patch(self, image):
        """
        Detect if adversarial patch is present in image using advanced defenses.
        
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
        
        # Process through advanced defense pipeline
        if self.use_advanced_defenses:
            processed_image = self.defense_pipeline.process_input(image, model=self.detection_model)
        else:
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
        
        # Also check visual similarity to reference patch if available
        visual_similarity = 0.0
        patch_match = False
        if self.patch is not None:
            try:
                # Compare input image to reference patch
                with torch.no_grad():
                    patch_output = self.detection_model(self.patch.unsqueeze(0) if len(self.patch.shape) == 3 else self.patch)
                    patch_probs = torch.softmax(patch_output, dim=1)
                    patch_pred = torch.argmax(patch_probs, dim=1)
                    patch_conf = torch.max(patch_probs, dim=1)[0]
                
                # Check if predictions match
                patch_match = (patch_pred == original_pred).item()
                
                # Compute visual similarity
                img_np = image[0].cpu().numpy() if len(image.shape) == 4 else image.cpu().numpy()
                patch_np = self.patch[0].cpu().numpy() if len(self.patch.shape) == 4 else self.patch.cpu().numpy()
                
                # Denormalize for comparison
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_denorm = np.clip((img_np * std + mean) * 255.0, 0, 255)
                patch_denorm = np.clip((patch_np * std + mean) * 255.0, 0, 255)
                
                # MSE-based similarity
                mse = np.mean((img_denorm - patch_denorm) ** 2)
                max_mse = 255.0 ** 2
                visual_similarity = 1.0 - (mse / max_mse)
                
            except Exception as e:
                logger.warning(f"Visual similarity calculation failed: {e}")
                visual_similarity = 0.0
        
        # Patch detected if any of these conditions are met:
        # 1. Patch effects (prediction change, confidence drop, low confidence)
        # 2. Visual similarity to reference patch (if patch is being scanned)
        # 3. Prediction match with reference patch
        patch_effects = pred_changed or conf_dropped or low_confidence
        visual_match = visual_similarity > 0.3  # 30% visual similarity threshold
        reference_match = patch_match and visual_similarity > 0.2  # Lower threshold for reference match
        
        is_detected = patch_effects or visual_match or reference_match
        
        # Log detection indicators for debugging
        logger.debug(f"Patch detection indicators - pred_changed: {pred_changed}, conf_dropped: {conf_dropped}, low_confidence: {low_confidence}")
        logger.debug(f"Confidence: {confidence.item() if isinstance(confidence, torch.Tensor) else confidence:.3f}, Original: {original_conf.item() if isinstance(original_conf, torch.Tensor) else original_conf:.3f}")
        
        # Validate through advanced defense pipeline
        prediction = {
            'class': pred_class.item() if isinstance(pred_class, torch.Tensor) else pred_class,
            'confidence': confidence.item() if isinstance(confidence, torch.Tensor) else confidence
        }
        
        # Use advanced defense validation
        import time
        validation = self.defense_pipeline.validate_detection(
            image,
            prediction,
            model=self.detection_model if self.use_advanced_defenses else None,
            frame_number=0,
            timestamp=time.time()
        )
        
        # Check if patch was detected by defenses
        patch_detected_by_defense = validation.get('patch_detected', False)
        
        # Defense bypassed ONLY if patch effect detected but defenses didn't catch it
        # This is a DEFENSE system - we want to detect and block patches
        defense_bypassed = is_detected and not patch_detected_by_defense
        
        # For a defense system, we want to report if a patch was detected (regardless of whether defense blocked it)
        # detected = True means "patch was present" (either blocked or bypassed)
        # patch_detected_by_defense = True means "defense successfully detected and blocked it"
        # defense_bypassed = True means "patch was present but defense failed to block it"
        detected = is_detected  # Report patch detection if any patch effect is detected
        
        # Log final detection result
        logger.info(f"Advanced Defense Detection - detected: {detected}, is_detected: {is_detected}, patch_detected_by_defense: {patch_detected_by_defense}, defense_bypassed: {defense_bypassed}")
        logger.info(f"Detection indicators - patch_effects: {patch_effects}, visual_similarity: {visual_similarity:.3f}, visual_match: {visual_match}, reference_match: {reference_match}")
        
        return {
            'detected': detected,
            'confidence': confidence.item() if isinstance(confidence, torch.Tensor) else confidence,
            'original_confidence': original_conf.item() if isinstance(original_conf, torch.Tensor) else original_conf,
            'defense_bypassed': defense_bypassed,
            'patch_detected_by_defense': patch_detected_by_defense,
            'defense_confidence': validation.get('detection_confidence', 0.0),
            'prediction': prediction,
            'prediction_changed': pred_changed,
            'confidence_dropped': conf_dropped,
            'visual_similarity': visual_similarity,
            'patch_match': patch_match,
            'validation_result': validation
        }
    
    def process_camera_feed(self, camera_id=0):
        """
        Process camera feed and detect patch with advanced defenses.
        
        Args:
            camera_id: Camera device ID
            command_type: Command to execute when patch detected
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        logger.info("Starting camera feed processing with Advanced Defenses...")
        logger.info("Advanced Defense System monitoring for adversarial patches")
        
        frame_count = 0
        
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
            
            # Normalize to ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            frame_tensor = (frame_tensor - mean) / std
            
            # Detect patch
            detection = self.detect_patch(frame_tensor)
            
            # Display detection status
            status_text = f"Confidence: {detection['confidence']:.3f}"
            if detection.get('patch_detected_by_defense', False):
                status_text += " - PATCH DETECTED BY DEFENSE!"
            elif detection['detected']:
                status_text += " - PATCH BYPASSED DEFENSES!"
            else:
                status_text += " - NO PATCH DETECTED"
            
            # Draw status on frame
            if detection.get('patch_detected_by_defense', False):
                # Defense caught it - show in green (success)
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "DEFENSE ACTIVE - ATTACK BLOCKED", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif detection['detected']:
                # Patch bypassed defenses - show in red (danger)
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "DEFENSE BYPASSED - ATTACK DETECTED", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # No patch detected
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if detection.get('patch_detected_by_defense', False):
                cv2.putText(frame, "PATCH DETECTED BY DEFENSE", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Advanced Defense System - Patch Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image_file(self, image_path):
        """
        Process a single image file with advanced defenses.
        
        Args:
            image_path: Path to image file
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
        print(f"Patch Detected by Defense: {detection.get('patch_detected_by_defense', False)}")
        print(f"Defense Confidence: {detection.get('defense_confidence', 0.0):.3f}")
        
        # Show defense results
        validation = detection.get('validation_result', {})
        defense_results = validation.get('defense_results', {})
        if defense_results:
            print("\nDefense Results:")
            if 'entropy' in defense_results:
                ent = defense_results['entropy']
                print(f"  Entropy Defense: {'DETECTED' if ent.get('patch_detected') else 'CLEAR'} (conf: {ent.get('confidence', 0.0):.3f})")
            if 'frequency' in defense_results:
                freq = defense_results['frequency']
                print(f"  Frequency Defense: {'DETECTED' if freq.get('patch_detected') else 'CLEAR'} (conf: {freq.get('confidence', 0.0):.3f})")
            if 'gradient_saliency' in defense_results:
                sal = defense_results['gradient_saliency']
                print(f"  Saliency Defense: {'DETECTED' if sal.get('patch_detected') else 'CLEAR'} (conf: {sal.get('confidence', 0.0):.3f})")
            if 'enhanced_multi_frame' in defense_results:
                mf = defense_results['enhanced_multi_frame']
                print(f"  Multi-Frame: {'CONSENSUS' if mf.get('consensus_reached') else 'NO CONSENSUS'}")
        
        logger.info(f"Image: {image_path}")
        logger.info(f"Confidence: {detection['confidence']:.3f}")
        logger.info(f"Patch Detected: {detection['detected']}")
        logger.info(f"Defense Bypassed: {detection['defense_bypassed']}")
        logger.info(f"Patch Detected by Defense: {detection.get('patch_detected_by_defense', False)}")
        
        # Report defense results
        if detection.get('patch_detected_by_defense', False):
            print("\n" + "="*70)
            print("DEFENSE SUCCESSFUL: Patch detected and blocked!")
            print("="*70)
            print("Adversarial patch attack PREVENTED by advanced defenses.")
            print("="*70)
            logger.info("DEFENSE SUCCESS: Patch detected by defenses, attack blocked")
        elif detection['detected'] and detection['defense_bypassed']:
            print("\n" + "="*70)
            print("WARNING: Patch bypassed all defenses!")
            print("="*70)
            print("Adversarial patch detected but defenses failed to block it.")
            print("="*70)
            logger.warning("DEFENSE FAILURE: Patch bypassed defenses")
        else:
            print("\nNo patch detected or patch effect not significant enough.")


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
    print("ADVANCED CYBERPHYSICAL DEFENSE SYSTEM - ADVERSARIAL PATCH DETECTION")
    print("="*70)
    print("\nThis system detects and blocks adversarial patches using Advanced Defense Pipeline")
    print("\nWARNING: This is for research/educational purposes only!")
    print("="*70)
    
    # Create attack patch image (without text)
    patch_path = 'data/patches/resnet_breaker_70pct.pt'
    if os.path.exists(patch_path):
        print(f"\nCreating attack patch image from: {patch_path}")
        image_path = create_attack_patch_image(patch_path, add_text=False)
        print(f"Attack patch image created: {image_path}")
        print("Note: Patch will trigger malware execution when detected!")
    else:
        print(f"\nPatch file not found: {patch_path}")
        print("Please train a patch first using train_resnet_breaker.py")
        sys.exit(1)
    
    # Initialize advanced defense system
    print("\nInitializing Advanced Cyberphysical Defense System...")
    print("Using Advanced Defense Pipeline with:")
    print("  - Entropy-Based Detection")
    print("  - Frequency Domain Analysis")
    print("  - Gradient Saliency Analysis")
    print("  - Enhanced Multi-Frame Smoothing")
    
    defense_system = AdvancedCyberphysicalAttackSystem(
        patch_path,
        use_advanced_defenses=True
    )
    
    print("\n" + "="*70)
    print("DEFENSE SYSTEM READY")
    print("="*70)
    print("\nOptions:")
    print("1. Process camera feed (real-time detection)")
    print("2. Process image file (test with patch image)")
    print("\nThis system detects and blocks adversarial patch attacks.")
    print("="*70)
    
    # For demo, process the patch image itself
    print("\nTesting with patch image...")
    defense_system.process_image_file(image_path)
    
    print("\n" + "="*70)
    print("To use camera feed, run:")
    print("  defense_system.process_camera_feed(camera_id=0)")
    print("="*70)

