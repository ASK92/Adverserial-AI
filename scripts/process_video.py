"""
Process video through the adversarial patch pipeline.
"""
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_loader import ModelLoader
from src.defenses.defense_pipeline import DefensePipeline, InputNormalization, AdversarialDetection, MultiFrameSmoothing, ContextRuleEngine
from src.patch.patch_applier import PatchApplier


def process_video(
    video_path: str,
    models: dict,
    defense_pipeline: DefensePipeline,
    output_path: str,
    device: str = 'cuda'
):
    """
    Process video through pipeline.
    
    Args:
        video_path: Input video path
        models: Dictionary of models
        defense_pipeline: Defense pipeline
        output_path: Output video path
        device: Computing device
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        
        # Process through defense pipeline
        processed_frame = defense_pipeline.process_input(frame_tensor)
        
        # Run inference (using first model)
        model_name = list(models.keys())[0]
        model = models[model_name]
        model.eval()
        
        with torch.no_grad():
            output = model(processed_frame)
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
            else:
                pred_class = 0
                confidence = 0.5
        
        # Validate through defense
        prediction = {
            'class': pred_class,
            'confidence': confidence
        }
        validation = defense_pipeline.validate_detection(
            processed_frame, prediction, frame_number=frame_count
        )
        
        # Draw results on frame
        if validation['valid']:
            detection_count += 1
            color = (0, 255, 0)  # Green
            text = f"Valid: Class {pred_class} ({confidence:.2f})"
        else:
            color = (0, 0, 255)  # Red
            text = f"Blocked: {validation.get('failed_defense', 'unknown')}"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, {detection_count} valid detections")
    
    cap.release()
    out.release()
    
    print(f"Processing complete: {output_path}")
    print(f"Total frames: {frame_count}, Valid detections: {detection_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video through pipeline')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--device', default='cuda', help='Computing device')
    
    args = parser.parse_args()
    
    # Load models
    model_loader = ModelLoader(device=args.device)
    models = {}
    
    try:
        models['resnet'] = model_loader.load_resnet('resnet50', pretrained=True)
    except:
        pass
    
    # Setup defense pipeline
    defense_pipeline = DefensePipeline(
        input_normalization=InputNormalization(enabled=True),
        adversarial_detection=AdversarialDetection(enabled=True, device=args.device),
        multi_frame_smoothing=MultiFrameSmoothing(enabled=True),
        context_rules=ContextRuleEngine(enabled=True),
        enabled=True
    )
    
    process_video(args.video, models, defense_pipeline, args.output, args.device)


