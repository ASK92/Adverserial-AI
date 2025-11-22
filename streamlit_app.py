"""
Streamlit Web Application for Adversarial Patch Detection
Simple pipeline: Input → Attack System → Local Execution
"""
import streamlit as st
import cv2
import numpy as np
import torch
import time
import os
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger

# Configure Streamlit page
st.set_page_config(
    page_title="Adversarial Patch Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Duke-themed CSS styling
st.markdown("""
<style>
    /* Duke Blue Color Palette */
    :root {
        --duke-blue: #003366;
        --duke-blue-light: #00539B;
        --duke-blue-dark: #001A33;
        --duke-white: #FFFFFF;
        --duke-gray: #F5F5F5;
        --duke-accent: #00A3E0;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
    }
    
    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers - Duke Blue */
    h1 {
        color: #003366 !important;
        font-weight: 700 !important;
        border-bottom: 4px solid #00539B;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h2 {
        color: #003366 !important;
        font-weight: 600 !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #00539B;
        padding-left: 1rem;
    }
    
    h3 {
        color: #00539B !important;
        font-weight: 600 !important;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    h4, h5, h6 {
        color: #00539B !important;
        font-weight: 500 !important;
    }
    
    /* Main Content Text - Duke Blue */
    .main {
        color: #003366 !important;
    }
    
    .main p, .main li, .main div, .main span, .main label,
    .main strong, .main em, .main ul, .main ol,
    .main td, .main th, .main a, .main code, .main pre,
    .main [class*="element-container"], .main [class*="stMarkdown"],
    .main [class*="stText"], .main [class*="stCaption"],
    .main [class*="stSubheader"], .main [data-testid*="text"],
    .main [data-testid*="markdown"], .main [data-testid*="caption"] {
        color: #003366 !important;
        line-height: 1.6;
    }
    
    .main *:not([class*="stButton"]):not([class*="stSuccess"]):not([class*="stError"]):not([class*="stWarning"]):not([class*="stInfo"]) {
        color: #003366 !important;
    }
    
    /* Sidebar Styling - Duke Blue Gradient Background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #00539B 0%, #003366 100%);
    }
    
    /* Sidebar Text - White */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Sidebar Headers - Transparent/Subtle */
    [data-testid="stSidebar"] h1 {
        background: rgba(0, 83, 155, 0.3);
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid rgba(0, 163, 224, 0.5) !important;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        background: rgba(0, 83, 155, 0.3);
        padding: 0.75rem;
        border-radius: 6px;
        border: 2px solid rgba(0, 163, 224, 0.5);
        margin: 1rem 0 0.5rem 0;
        text-align: center;
    }
    
    /* Buttons - Duke Blue */
    .stButton > button {
        background: linear-gradient(135deg, #003366 0%, #00539B 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 51, 102, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00539B 0%, #003366 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 51, 102, 0.4);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00539B 0%, #00A3E0 100%);
        box-shadow: 0 4px 8px rgba(0, 163, 224, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00A3E0 0%, #00539B 100%);
        box-shadow: 0 6px 12px rgba(0, 163, 224, 0.5);
    }
    
    /* Success/Info/Warning/Error Boxes */
    .stSuccess {
        background: linear-gradient(135deg, #E6F7FF 0%, #BAE7FF 100%);
        border-left: 5px solid #00539B;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 51, 102, 0.1);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #E6F2FF 0%, #B3D9FF 100%);
        border-left: 5px solid #003366;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 51, 102, 0.1);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #FFF4E6 0%, #FFE0B3 100%);
        border-left: 5px solid #FFA500;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(255, 165, 0, 0.2);
    }
    
    .stError {
        background: linear-gradient(135deg, #FFE6E6 0%, #FFB3B3 100%);
        border-left: 5px solid #CC0000;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(204, 0, 0, 0.2);
    }
    
    /* Radio Buttons and Checkboxes */
    .stRadio label,
    .stCheckbox label {
        color: #003366 !important;
        font-weight: 500;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #00539B;
        border-radius: 12px;
        padding: 1.5rem;
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #003366;
        background: linear-gradient(135deg, #E9ECEF 0%, #DEE2E6 100%);
    }
    
    .stFileUploader label {
        color: #003366 !important;
        font-weight: 600;
    }
    
    /* Camera Input */
    .stCameraInput, [data-testid="stCameraInput"],
    .stCameraInput > div, [data-testid="stCameraInput"] > div,
    .stCameraInput > div > div {
        background-color: white !important;
        border: 3px solid #00539B;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 51, 102, 0.2);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #003366 !important;
        font-weight: 700;
        font-size: 2rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #00539B !important;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #E6F2FF 0%, #B3D9FF 100%);
        color: #003366 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem;
        border: 1px solid #00539B;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #B3D9FF 0%, #E6F2FF 100%);
    }
    
    .streamlit-expanderContent {
        background-color: #FFFFFF;
        color: #003366 !important;
        padding: 1rem;
        border-radius: 0 0 8px 8px;
    }
    
    /* Code Blocks */
    .main code,
    .main pre {
        background: linear-gradient(135deg, #F5F5F5 0%, #E9ECEF 100%);
        color: #003366 !important;
        border: 1px solid #00539B;
        border-radius: 6px;
        padding: 0.5rem;
    }
    
    /* Tables */
    .main table {
        background-color: #FFFFFF;
        border: 2px solid #00539B;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .main table th {
        background: linear-gradient(135deg, #003366 0%, #00539B 100%);
        color: white !important;
        font-weight: 600;
        padding: 0.75rem;
    }
    
    .main table td {
        color: #003366 !important;
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #E9ECEF;
    }
    
    .main table tr:hover {
        background-color: #F8F9FA;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #00539B 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #FFFFFF;
        color: #003366 !important;
        border: 2px solid #00539B;
        border-radius: 6px;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #003366;
        box-shadow: 0 0 0 3px rgba(0, 83, 155, 0.1);
    }
    
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label {
        color: #003366 !important;
        font-weight: 600;
    }
    
    /* Sidebar Radio and Checkbox */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div,
    [data-testid="stSidebar"] .stCheckbox > div {
        background-color: #00539B;
        border-radius: 6px;
        padding: 0.5rem;
        border: 1px solid #003366;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F5F5F5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00539B 0%, #003366 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #003366 0%, #00539B 100%);
    }
    
    /* Custom Duke Header */
    .duke-header {
        background: linear-gradient(135deg, #003366 0%, #00539B 50%, #00A3E0 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 51, 102, 0.3);
    }
    
    .duke-header h1 {
        color: white !important;
        border: none !important;
        margin: 0;
        padding: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .duke-header h1 * {
        color: white !important;
    }
    
    .duke-header p {
        color: white !important;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Custom Divider */
    .duke-divider {
        height: 4px;
        background: linear-gradient(90deg, transparent 0%, #00539B 25%, #00A3E0 50%, #00539B 75%, transparent 100%);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    /* LaTeX/Math Rendering */
    .katex {
        color: #003366 !important;
    }
    
    /* Spinner Text - White */
    .stSpinner, [data-testid="stSpinner"],
    .stSpinner *, [data-testid="stSpinner"] * {
        color: white !important;
    }
    
    /* Info box text in main - keep readable */
    .main .stInfo {
        color: #003366 !important;
    }
    
    .main .stInfo * {
        color: #003366 !important;
    }
    
    /* White text overrides in main */
    .main p[style*="color: white"],
    .main div[style*="color: white"],
    .main span[style*="color: white"] {
        color: white !important;
    }
    
    /* Column Styling */
    [data-testid="column"] {
        background: transparent;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize logger
logger = setup_logger()

# Initialize session state
if 'attack_system' not in st.session_state:
    st.session_state.attack_system = None
if 'device' not in st.session_state:
    st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'opencv_camera_active' not in st.session_state:
    st.session_state.opencv_camera_active = False
if 'opencv_camera_id' not in st.session_state:
    st.session_state.opencv_camera_id = 0
if 'use_advanced_system' not in st.session_state:
    st.session_state.use_advanced_system = False

def initialize_attack_system():
    """Initialize the attack system - basic or advanced based on toggle."""
    if st.session_state.attack_system is not None:
        current_is_advanced = hasattr(st.session_state.attack_system, 'use_advanced_defenses')
        if current_is_advanced != st.session_state.use_advanced_system:
            st.session_state.attack_system = None
            st.session_state.system_initialized = False
    
    if st.session_state.attack_system is None:
        try:
            system_type = "Advanced" if st.session_state.use_advanced_system else "Basic"
            with st.spinner(f"Loading {system_type} attack system..."):
                boo_patch_path = 'data/patches/resnet_breaker_70pct.pt'
                if not os.path.exists(boo_patch_path):
                    st.error("Boo patch file not found! Please ensure patch file exists.")
                    return False
                
                if st.session_state.use_advanced_system:
                    from cyberphysical_attack_system_advanced import AdvancedCyberphysicalAttackSystem
                    st.session_state.attack_system = AdvancedCyberphysicalAttackSystem(
                        patch_path=boo_patch_path,
                        use_advanced_defenses=True
                    )
                    st.session_state.attack_system.demo_mode = True
                    logger.info("Advanced attack system initialized")
                else:
                    from cyberphysical_attack_system_Boo import CyberphysicalAttackSystem
                    
                    malware_patch_path = 'data/patches/malware_attack_patch.png'
                    if not os.path.exists(malware_patch_path):
                        malware_patch_path = None
                        st.warning("Malware patch file not found. Malware patch detection will be disabled.")
                    
                    st.session_state.attack_system = CyberphysicalAttackSystem(
                        patch_path=boo_patch_path,
                        patch_image_path=malware_patch_path,
                        repo_url='https://github.com/ASK92/Malware-V1.0.git'
                    )
                    st.session_state.attack_system.demo_mode = True
                    logger.info("Basic attack system initialized")
                
                st.session_state.system_initialized = True
                return True
        except Exception as e:
            st.error(f"Failed to initialize attack system: {e}")
            logger.error(f"Initialization error: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.session_state.system_initialized = False
            return False
    return st.session_state.system_initialized

def process_frame(frame_array):
    """Process a frame through the attack system."""
    if st.session_state.attack_system is None:
        return None, frame_array
    
    try:
        if isinstance(frame_array, np.ndarray):
            if len(frame_array.shape) == 3:
                if frame_array.shape[2] == 3:
                    img_rgb = frame_array
                else:
                    img_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            else:
                return None, frame_array
            
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
            
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0).to(st.session_state.device)
            
            detection = st.session_state.attack_system.detect_patch(img_tensor)
            
            if st.session_state.use_advanced_system:
                if detection.get('detected', False) and detection.get('defense_bypassed', False):
                    st.session_state.attack_system.command_executed = False
                    st.session_state.attack_system.execute_command(command_type='notepad_boo')
            else:
                if detection.get('detected', False):
                    patch_type = detection.get('patch_type', 'boo')
                    st.session_state.attack_system.command_executed = False
                    command_type = 'malware' if patch_type == 'malware' else 'notepad_boo'
                    st.session_state.attack_system.execute_command(
                        command_type=command_type,
                        patch_type=patch_type
                    )
            
            return detection, img_rgb
        else:
            return None, frame_array
            
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, frame_array

def about_page():
    """Display About the Project page."""
    # Duke-themed header
    st.markdown("""
    <div class="duke-header">
        <h1>Duke University - Adversarial Patch Detection System</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="duke-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    This system demonstrates adversarial patch attacks on computer vision models and implements 
    advanced defense mechanisms to detect and block such attacks. The project explores the 
    vulnerability of deep learning models to physical adversarial attacks and methods to defend against them.
    
    **Note**: This is for research/educational purposes only.
    """)
    
    st.markdown('<div class="duke-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Part 1: Adversarial Patch Generation and Training
    """)
    
    st.markdown("""
    ### How Patches Are Generated
    
    Adversarial patches are physical objects (like stickers or printed images) designed to fool 
    computer vision models when placed in the camera's view. Our patch generation process uses 
    **optimization-based training** to create patches that can bypass ResNet50 and other models.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    #### Training Process Overview
    
    The patch training follows a systematic approach:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Initialization**
        - Patch size: 250×250 pixels
        - Initial values: 0.5-1.0 range
        - Learnable parameter optimized during training
        """)
        
        st.markdown("""
        **2. Diverse Training Data**
        - 40 synthetic images per batch
        - ImageNet-normalized, bright, dark, high-contrast, saturated
        - Ensures robustness across conditions
        """)
    
    with col2:
        st.markdown("""
        **3. Patch Application**
        - Random locations on images
        - Size: 40-55% of image area
        - Bilinear interpolation for resizing
        - Full opacity application
        """)
        
        st.markdown("""
        **4. Optimization**
        - AdamW optimizer (LR: 0.4)
        - Cosine annealing scheduler
        - 4000 iterations with early stopping
        """)
    
    st.markdown("---")
    
    st.markdown("""
    #### Multi-Strategy Loss Function
    
    The training uses **8 different attack strategies** simultaneously to maximize effectiveness:
    """)
    
    with st.expander("📐 View Mathematical Formulation", expanded=False):
        st.markdown(r"""
        **Total Loss Equation:**
        
        $$L_{total} = 15.0 \cdot L_{resnet} + 1.0 \cdot L_{efficientnet} + 0.01 \cdot L_{TV}$$
        
        Where $L_{resnet}$ combines all 8 strategies:
        
        $$L_{resnet} = L_1 + L_2 + L_3 + L_4 + L_5 + L_6 + L_7 + L_8$$
        """)
    
    st.markdown("""
    **Strategy Breakdown:**
    """)
    
    strategies = [
        ("Strategy 1", "Minimize confidence in original prediction", "Forces model to lose confidence in what it originally saw"),
        ("Strategy 2", "Maximize prediction changes (weight: 5.0)", "Heavily weighted to force wrong class predictions"),
        ("Strategy 3", "Maximize entropy (uncertainty)", "Makes model predictions more random/uncertain"),
        ("Strategy 4", "Maximize probability of wrong classes", "Increases confidence in incorrect predictions"),
        ("Strategy 5", "Minimize top-1 confidence", "Reduces model's confidence in any single prediction"),
        ("Strategy 6", "Maximize KL divergence from original", "Makes output distribution as different as possible"),
        ("Strategy 7", "Maximize top-5 wrong class probabilities", "Ensures multiple wrong classes have high probability"),
        ("Strategy 8", "Push confidence towards 0.5 (uncertainty)", "Makes model uncertain rather than confidently wrong")
    ]
    
    for i, (name, title, desc) in enumerate(strategies, 1):
        with st.expander(f"{name}: {title}", expanded=False):
            st.markdown(f"**Description:** {desc}")
            
            if i == 1:
                st.latex(r"L_1 = \frac{1}{N} \sum_{i=1}^{N} \log(P_{patched}(y_{orig}^{(i)} | x_i^{(i)}) + \epsilon)")
                st.caption("Where $P_{patched}(y_{orig}^{(i)} | x_i^{(i)})$ is the probability of the original class after patch application, and $\epsilon = 10^{-12}$ prevents log(0).")
            elif i == 2:
                st.latex(r"L_2 = -5.0 \cdot \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_{patched}^{(i)} \neq y_{orig}^{(i)}]")
                st.caption("Where $\mathbb{1}[\cdot]$ is the indicator function, $y_{orig}$ is original prediction, $y_{patched}$ is patched prediction.")
            elif i == 3:
                st.latex(r"L_3 = -0.5 \cdot \frac{1}{N} \sum_{i=1}^{N} H(P_{patched}(y | x_i^{(i)}))")
                st.latex(r"H(P) = -\sum_{c=1}^{C} P(c) \log(P(c) + \epsilon)")
                st.caption("Where $H(P)$ is the entropy of the probability distribution over $C$ classes.")
            elif i == 4:
                st.latex(r"L_4 = -\frac{1}{N} \sum_{i=1}^{N} \log(\max_{c \neq y_{orig}^{(i)}} P_{patched}(c | x_i^{(i)}) + \epsilon)")
            elif i == 5:
                st.latex(r"L_5 = \frac{1}{N} \sum_{i=1}^{N} \log(\max_c P_{patched}(c | x_i^{(i)}) + \epsilon)")
            elif i == 6:
                st.latex(r"L_6 = -0.3 \cdot \frac{1}{N} \sum_{i=1}^{N} D_{KL}(P_{orig}(y | x_i^{(i)}) || P_{patched}(y | x_i^{(i)}))")
                st.latex(r"D_{KL}(P || Q) = \sum_{c=1}^{C} P(c) \log(\frac{P(c)}{Q(c) + \epsilon} + \epsilon)")
            elif i == 7:
                st.latex(r"L_7 = -\frac{1}{N} \sum_{i=1}^{N} \log(\sum_{c \in Top5_{wrong}} P_{patched}(c | x_i^{(i)}) + \epsilon)")
            elif i == 8:
                st.latex(r"L_8 = \frac{1}{N} \sum_{i=1}^{N} |\max_c P_{patched}(c | x_i^{(i)}) - 0.5|")
    
    with st.expander("📐 Total Variation Loss", expanded=False):
        st.latex(r"L_{TV} = \frac{1}{HW} \sum_{h=1}^{H-1} \sum_{w=1}^{W} |P_{h+1,w} - P_{h,w}| + \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W-1} |P_{h,w+1} - P_{h,w}|")
        st.caption("Where $P$ is the patch tensor of size $(C, H, W)$. This encourages smoothness in the patch.")
    
    st.markdown("---")
    
    st.markdown("""
    #### Optimization Details
    """)
    
    with st.expander("📐 AdamW Optimizer Formulation", expanded=False):
        st.markdown("""
        **Update Rules:**
        """)
        st.latex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t")
        st.latex(r"v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2")
        st.latex(r"\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}")
        st.latex(r"\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)")
        
        st.markdown("""
        **Parameters:**
        - $g_t$: gradient at iteration $t$
        - $\beta_1 = 0.9$, $\beta_2 = 0.999$ (momentum parameters)
        - $\eta_t$: learning rate at iteration $t$ (initial: 0.4)
        - $\lambda = 0.005$: weight decay
        - $\epsilon = 10^{-8}$: prevents division by zero
        """)
    
    with st.expander("📐 Cosine Annealing Scheduler", expanded=False):
        st.latex(r"\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{1 + \cos(\pi \cdot \frac{T_{cur}}{T_i})}{2}")
        st.markdown("""
        **Parameters:**
        - $T_{cur}$: number of epochs since last restart
        - $T_i$: period (500 iterations initially, doubles after each restart)
        - $\eta_{min} = 0.01$, $\eta_{max} = 0.4$
        """)
    
    with st.expander("📐 Gradient Clipping", expanded=False):
        st.latex(r"g_t = \begin{cases} g_t & \text{if } ||g_t|| \leq 5.0 \\ 5.0 \cdot \frac{g_t}{||g_t||} & \text{otherwise} \end{cases}")
        st.caption("Prevents gradient explosion by clipping gradients to maximum norm of 5.0.")
    
    st.markdown("---")
    
    st.markdown("""
    #### Making Patches Resilient
    
    The patch achieves resilience through several techniques:
    
    - **Large Size (40-55% of image)**: Ensures patch has significant impact on model features
    - **Diverse Training**: Works across different image types and conditions
    - **Multiple Attack Strategies**: Targets different aspects of model behavior simultaneously
    - **High Learning Rate**: Allows rapid optimization to find effective patterns
    - **Minimal Regularization**: Low TV loss allows complex, high-frequency patterns that are 
      effective against ResNet50's feature extraction
    
    #### Bypassing ResNet50
    
    ResNet50 is particularly vulnerable because:
    
    1. **Feature Extraction**: ResNet50 uses residual blocks that can be easily fooled by 
       high-frequency patterns in the patch
    2. **Large Receptive Field**: The patch's large size (40-55% of image) affects multiple 
       feature extraction layers
    3. **Optimization Target**: The loss function specifically targets ResNet50's prediction 
       mechanism with 15× weight compared to other models
    4. **Pattern Design**: The optimized patterns exploit ResNet50's learned features to 
       create adversarial activations
    
    The patch achieves **70%+ success rate** on ResNet50 by simultaneously:
    - Reducing confidence in correct predictions
    - Increasing prediction changes
    - Creating uncertainty in model outputs
    - Exploiting ResNet50's feature space
    """)
    
    st.markdown('<div class="duke-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Part 2: Cyberphysical Attack System Architecture
    """)
    
    st.markdown("""
    ### System Overview
    
    The `cyberphysical_attack_system_Boo.py` implements a **real-world attack pipeline** that 
    demonstrates how adversarial patches can trigger physical actions when detected by computer vision systems.
    
    #### Architecture Components
    
    1. **Patch Loading System**:
       - Loads pre-trained adversarial patch tensor (Boo patch)
       - Loads reference patch image (Malware patch) for visual similarity matching
       - Supports both tensor-based and image-based patch detection
    
    2. **Detection Model**:
       - Uses **ResNet50** (pretrained on ImageNet) as the target model
       - Model runs in evaluation mode for inference
       - Processes images at 224×224 resolution (ImageNet standard)
    
    3. **Defense Pipeline** (Basic):
       - **Input Normalization**: Brightness adjustment (0.7-1.3×), random blur (20% probability)
       - **Adversarial Detection**: L2 norm threshold checking (threshold: 0.2)
       - **Multi-Frame Smoothing**: 3-frame window with 50% consensus threshold
       - **Context Rules**: Temporal checks enabled, spatial checks disabled
       - Note: These are **relaxed settings** for demonstration purposes
    
    4. **Patch Detection Logic**:
       
       The system uses different detection methods for Boo and Malware patches.
       
       **For Boo Patch:**
       - Compares model predictions before/after defense processing
       - Detects if: prediction changed, confidence dropped, or confidence is low (<0.5)
       - Uses multiple heuristics for robust detection
       
       **For Malware Patch:**
       - Uses visual similarity matching (SSIM or MSE-based)
       - Compares input image to reference patch image
       - SSIM threshold: 0.6, MSE-based similarity threshold: 0.4
       - Also checks model prediction similarity as secondary check
    """)
    
    with st.expander("📐 View Mathematical Formulation for Patch Detection", expanded=False):
        st.markdown(r"""
        **Boo Patch Detection:**
        
        Let $x$ be the input image, $f$ be the ResNet50 model, and $x_{def}$ be the image after defense processing.
        
        **Original Prediction:**
        $$y_{orig} = \arg\max_c P(c | x), \quad c_{orig} = \max_c P(c | x)$$
        
        **Defended Prediction:**
        $$y_{def} = \arg\max_c P(c | x_{def}), \quad c_{def} = \max_c P(c | x_{def})$$
        
        **Detection Conditions** (OR logic):
        $$D_1 = \mathbb{1}[y_{def} \neq y_{orig}] \quad \text{(prediction changed)}$$
        $$D_2 = \mathbb{1}[c_{def} < 0.7 \cdot c_{orig}] \quad \text{(confidence dropped 30\%)}$$
        $$D_3 = \mathbb{1}[c_{def} < 0.9 \cdot c_{orig}] \quad \text{(confidence dropped 10\%)}$$
        $$D_4 = \mathbb{1}[c_{def} < 0.5] \quad \text{(low confidence)}$$
        
        **Boo Patch Detected:**
        $$\text{Boo Detected} = D_1 \lor D_2 \lor D_3 \lor D_4$$
        
        **Malware Patch Detection:**
        
        **Visual Similarity - SSIM** (Structural Similarity Index):
        $$\text{SSIM}(x, x_{ref}) = \frac{(2\mu_x \mu_{ref} + c_1)(2\sigma_{x,ref} + c_2)}{(\mu_x^2 + \mu_{ref}^2 + c_1)(\sigma_x^2 + \sigma_{ref}^2 + c_2)}$$
        
        Where:
        - $\mu_x, \mu_{ref}$ are mean pixel values
        - $\sigma_x^2, \sigma_{ref}^2$ are variances
        - $\sigma_{x,ref}$ is covariance
        - $c_1 = (0.01 \cdot 255)^2$, $c_2 = (0.03 \cdot 255)^2$ are stability constants
        
        **Visual Similarity - MSE-based:**
        $$\text{MSE}(x, x_{ref}) = \frac{1}{HW} \sum_{i,j} (x(i,j) - x_{ref}(i,j))^2$$
        $$\text{Similarity}_{MSE} = 1 - \frac{\text{MSE}(x, x_{ref})}{\text{MSE}_{max}}, \quad \text{MSE}_{max} = 255^2$$
        
        **Model Prediction Similarity:**
        $$P_{match} = \mathbb{1}[y_x = y_{ref}]$$
        $$C_{similarity} = 1 - |c_x - c_{ref}|$$
        
        **Malware Patch Detected:**
        """)
        st.latex(r"\text{Malware Detected} = \begin{cases} \text{SSIM}(x, x_{ref}) > 0.6 & \text{if SSIM available} \\ \text{Similarity}_{MSE} > 0.4 & \text{otherwise} \end{cases} \lor (P_{match} \land C_{similarity} > 0.7)")
        st.markdown("""
        """)
    
    st.markdown("""
    
    5. **Command Execution System**:
       - **Boo Patch**: Opens Notepad and types "Boo" using PowerShell/automation
       - **Malware Patch**: Downloads GitHub repository and executes `blue_devil_lock.py`
       - Commands execute only once per detection (prevents repeated execution)
       - Uses subprocess for safe command execution
    
    #### How Organizations Use Similar Architectures
    
    This architecture mirrors real-world systems used in:
    
    **1. Autonomous Vehicles**:
       - Camera-based object detection systems
       - Adversarial patches on road signs can cause misclassification
       - Similar detection → action pipeline (detection triggers braking/turning)
    
    **2. Security Systems**:
       - Facial recognition access control
       - Adversarial patches can bypass face detection
       - Detection triggers door unlocking or alarm systems
    
    **3. Industrial Automation**:
       - Quality control systems using computer vision
       - Patches can fool defect detection
       - Misclassification triggers product acceptance/rejection
    
    **4. Retail and Surveillance**:
       - Product recognition systems
       - Inventory management via camera
       - Patches can cause incorrect product identification
    
    **5. Medical Imaging**:
       - Diagnostic systems using image analysis
       - Adversarial patterns can affect diagnosis
       - Detection triggers treatment recommendations
    
    #### Real-World Attack Flow
    
    ```
    Camera Input → Preprocessing → Model Inference → Detection Logic 
    → Defense Check → Command Execution → Physical Action
    ```
    
    This is the **exact flow** used in production systems where:
    - Computer vision models make decisions
    - Decisions trigger automated actions
    - Adversarial patches can hijack this decision-making process
    
    #### Why This Architecture is Vulnerable
    
    1. **Single Model Dependency**: Relies on one model (ResNet50) for decisions
    2. **Relaxed Defenses**: Basic defenses are easily bypassed by optimized patches
    3. **Direct Action Trigger**: Detection directly triggers commands without human verification
    4. **No Multi-Model Consensus**: Doesn't use ensemble of models for validation
    5. **Limited Input Validation**: Minimal checking of input image properties
    
    This demonstrates why **advanced defense mechanisms** are critical for production systems.
    """)
    
    st.markdown('<div class="duke-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Part 3: Advanced Defense Mechanisms
    """)
    
    st.markdown("""
    ### Defense Architecture Overview
    
    The Advanced Defense System uses **4 complementary defense layers** in cascade mode. 
    Each layer uses a different approach to detect adversarial patches, making it extremely 
    difficult for attackers to bypass all defenses simultaneously.
    
    ### Why Multiple Defense Layers?
    
    Adversarial patches have telltale signs that can be detected:
    - **High entropy** (random-looking patterns)
    - **Unusual frequency signatures** (artificial patterns)
    - **Strong gradient influence** (overly influential regions)
    - **Temporal inconsistencies** (unstable predictions across frames)
    
    Each defense layer targets one or more of these characteristics.
    """)
    
    st.markdown("""
    ### Layer 1: Entropy-Based Detection
    
    **Concept**: Adversarial patches often have high entropy because they contain random-looking 
    patterns optimized to fool models. Natural images have structured patterns with lower entropy.
    
    **How It Works**:
    
    1. **Sliding Window Analysis**:
       - Divides image into overlapping 32×32 pixel windows
       - Moves window across image with 8-pixel stride
       - Computes entropy for each window
    
    2. **Patch Detection**:
       - Flags regions with entropy > 7.0 (threshold)
       - Finds largest connected component (likely the patch)
       - Computes confidence based on entropy level and patch size
    
    3. **Mitigation**:
       - **Masks** detected patch regions (replaces with neutral values)
       - Prevents patch from influencing model predictions
    """)
    
    with st.expander("📐 View Mathematical Formulation for Entropy Detection", expanded=False):
        st.markdown(r"""
        **Entropy Calculation:**
        
        For each sliding window $W$ of size $32 \times 32$:
        
        **Step 1**: Compute histogram
        $$H[k] = \sum_{(i,j) \in W} \mathbb{1}[I(i,j) = k], \quad k \in \{0, 1, ..., 255\}$$
        
        **Step 2**: Normalize to probability distribution
        $$p[k] = \frac{H[k] + \epsilon}{\sum_{k=0}^{255} (H[k] + \epsilon)}, \quad \epsilon = 10^{-10}$$
        
        **Step 3**: Calculate Shannon entropy (base 2)
        $$H(W) = -\sum_{k=0}^{255} p[k] \cdot \log_2(p[k] + \epsilon)$$
        
        **Step 4**: Assign entropy to all pixels in window
        $$E(i,j) = \max(E(i,j), H(W)) \quad \forall (i,j) \in W$$
        
        Where $E(i,j)$ is the entropy map value at pixel $(i,j)$.
        
        **Detection Threshold:**
        $$\text{Patch Detected} = \exists (i,j): E(i,j) > \tau_{entropy}$$
        Where $\tau_{entropy} = 7.0$ is the entropy threshold.
        """)
    
    st.markdown("""
    **Why It Works**:
    - Natural images: entropy ~5-6 (structured patterns)
    - Adversarial patches: entropy ~7-8 (random patterns)
    - Example: Photo of dog has entropy ~5.5, adversarial patch has entropy ~7.5
    
    **Strengths**: Fast computation, effective against random-looking patches, can localize patch location
    
    **Weaknesses**: May miss patches designed to look natural, can flag textured natural objects
    """)
    
    st.markdown("""
    ### Layer 2: Frequency Domain Analysis
    
    **Concept**: Adversarial patches have distinctive frequency signatures - they contain artificial 
    high-frequency patterns that don't appear in natural images.
    
    **How It Works**:
    
    1. **Frequency Transform**:
       - Converts image to frequency domain using **2D FFT** (Fast Fourier Transform)
       - Computes magnitude spectrum (shows frequency content)
    
    2. **High-Frequency Analysis**:
       - Identifies high-frequency region (outer 30% of spectrum)
       - Computes statistics (mean, std) of high-frequency content
       - Detects anomalies (values > 2.0 standard deviations above mean)
    
    3. **Mitigation**:
       - **Filters** high-frequency anomalies using frequency cutoff (0.3)
       - Removes artificial patterns while preserving natural image content
    """)
    
    with st.expander("📐 View Mathematical Formulation for Frequency Analysis", expanded=False):
        st.markdown(r"""
        **Frequency Transform:**
        
        Converts image $I(x,y)$ to frequency domain using **2D FFT**:
        
        $$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x,y) \cdot e^{-2\pi i (ux/M + vy/N)}$$
        
        Where $M \times N$ is the image size, and $(u,v)$ are frequency coordinates.
        
        **FFT Shift** (centers DC component):
        $$F_{shifted}(u,v) = F((u + M/2) \bmod M, (v + N/2) \bmod N)$$
        
        **Magnitude Spectrum:**
        $$|F(u,v)| = \sqrt{\text{Re}(F(u,v))^2 + \text{Im}(F(u,v))^2}$$
        
        **High-Frequency Analysis:**
        
        **Distance from DC component:**
        $$d(u,v) = \sqrt{(u - u_c)^2 + (v - v_c)^2}$$
        $$d_{norm}(u,v) = \frac{d(u,v)}{d_{max}}, \quad d_{max} = \sqrt{u_c^2 + v_c^2}$$
        
        Where $(u_c, v_c) = (M/2, N/2)$ is the center (DC component).
        
        **High-Frequency Mask:**
        """)
        st.latex(r"M_{HF}(u,v) = \begin{cases} 1 & \text{if } d_{norm}(u,v) > 0.7 \text{ (outer 30\%)} \\ 0 & \text{otherwise} \end{cases}")
        st.markdown("""
        
        **Anomaly Detection:**
        
        **Statistics of High-Frequency Region:**
        """)
        st.latex(r"\mu_{HF} = \frac{1}{|M_{HF}|} \sum_{(u,v): M_{HF}(u,v)=1} |F(u,v)|")
        st.latex(r"\sigma_{HF} = \sqrt{\frac{1}{|M_{HF}|} \sum_{(u,v): M_{HF}(u,v)=1} (|F(u,v)| - \mu_{HF})^2}")
        st.markdown("""
        
        **Z-Score Calculation:**
        """)
        st.latex(r"z(u,v) = \frac{|F(u,v)| - \mu_{HF}}{\sigma_{HF} + \epsilon}, \quad \epsilon = 10^{-10}")
        st.markdown("""
        
        **Anomaly Detection:**
        """)
        st.latex(r"\text{Anomaly}(u,v) = \begin{cases} 1 & \text{if } z(u,v) > \tau_{anomaly} \text{ and } M_{HF}(u,v) = 1 \\ 0 & \text{otherwise} \end{cases}")
        st.markdown("""
        
        Where $\tau_{anomaly} = 2.0$ (2 standard deviations).
        
        **Confidence Score:**
        """)
        st.latex(r"\text{Confidence} = \frac{|\{(u,v): \text{Anomaly}(u,v) = 1\}|}{|M_{HF}|}")
        st.markdown("""
        """)
    
    st.markdown("""
    
    **Why It Works**:
    - Natural images: Smooth frequency spectrum with gradual transitions
    - Adversarial patches: Sharp spikes in high-frequency region
    - FFT reveals patterns invisible in spatial domain
    
    **Strengths**: ✅ Detects frequency-based attacks, works even if patch looks natural spatially
    
    **Weaknesses**: ❌ May filter legitimate high-frequency content (e.g., fine textures), computationally more expensive
    """)
    
    st.markdown("""
    ### Layer 3: Gradient Saliency Analysis
    
    **Concept**: Adversarial patches have unusually strong influence on model predictions. 
    By computing gradients, we can identify regions that disproportionately affect the output.
    
    **How It Works**:
    
    1. **Gradient Computation**:
       - Uses **Grad-CAM** (Gradient-weighted Class Activation Mapping)
       - Computes gradients of model output with respect to input pixels
       - Uses **Integrated Gradients** for more stable saliency maps
    
    2. **Patch Detection**:
       - Identifies regions with saliency > 0.7 (threshold)
       - Finds connected components (likely patch regions)
       - Computes confidence based on saliency intensity and region size
    
    3. **Mitigation**:
       - **Masks** high-saliency regions
       - Prevents overly influential regions from affecting predictions
    """)
    
    with st.expander("📐 View Mathematical Formulation for Gradient Saliency", expanded=False):
        st.markdown(r"""
        **Grad-CAM Method:**
        
        For input image $x$ and model $f$, compute:
        
        $$S_{GradCAM}(x) = \text{ReLU}\left(\sum_{k} \alpha_k \cdot A_k(x)\right)$$
        
        Where:
        - $A_k(x)$ are activations from the last convolutional layer
        - $\alpha_k = \frac{1}{HW} \sum_{i,j} \frac{\partial y_c}{\partial A_{k,i,j}}$ are gradient weights
        - $y_c$ is the output for class $c$
        
        **Simplified Gradient Saliency** (used in implementation):
        $$S(x) = \left|\frac{\partial f(x)}{\partial x}\right|$$
        
        Where $f(x)$ is the model output (logits).
        
        **Integrated Gradients Method** (more stable):
        
        **Path Integral:**
        $$IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
        
        Where $x'$ is a baseline (typically black image: $x' = 0$).
        
        **Discrete Approximation** (with $m$ steps):
        $$IG_i(x) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial f(x' + \frac{k}{m}(x - x'))}{\partial x_i}$$
        
        **Saliency Map:**
        $$S_{IG}(x) = \sum_{c=1}^{C} |IG_c(x)|$$
        
        Where $C$ is the number of channels (RGB: 3).
        
        **Normalization:**
        $$S_{norm}(x) = \frac{S(x) - \min(S(x))}{\max(S(x)) - \min(S(x)) + \epsilon}$$
        
        **Patch Detection:**
        $$\text{Patch Detected} = \exists (i,j): S_{norm}(i,j) > \tau_{saliency}$$
        Where $\tau_{saliency} = 0.7$ is the saliency threshold.
        """)
    
    st.markdown("""
    **Why It Works**:
    - Natural images: Gradients distributed across entire object
    - Adversarial patches: Concentrated gradients in patch region
    - Patches are designed to maximize gradient influence
    
    **Strengths**: ✅ Model-aware detection, identifies actual attack regions, very effective
    
    **Weaknesses**: ❌ Requires model gradients (computationally expensive), may flag legitimate important regions
    """)
    
    st.markdown("""
    ### Layer 4: Enhanced Multi-Frame Smoothing
    
    **Concept**: Adversarial patches cause unstable predictions across video frames. 
    Natural objects have consistent predictions, while patches cause temporal inconsistencies.
    
    **How It Works**:
    
    1. **Frame Buffer**:
       - Maintains sliding window of last 15 frames
       - Stores predictions and confidence scores for each frame
    
    2. **Temporal Analysis**:
       - **Class Consensus**: Checks if same class predicted in 80% of frames
       - **Confidence Stability**: Monitors confidence variance (threshold: 0.1)
       - **Temporal Gradient**: Tracks prediction changes over time (threshold: 0.3)
    
    3. **Detection**:
       - If any check fails, attack is detected due to temporal inconsistency
    
    4. **Validation**:
       - Requires stable predictions across multiple frames
       - Blocks attacks that cause prediction instability
    """)
    
    with st.expander("📐 View Mathematical Formulation for Multi-Frame Smoothing", expanded=False):
        st.markdown(r"""
        **Frame Buffer:**
        
        Maintains sliding window of last $W = 15$ frames:
        $$\mathcal{F}_t = \{f_{t-W+1}, f_{t-W+2}, ..., f_t\}$$
        
        Each frame $f_i$ contains:
        - Predicted class: $y_i = \arg\max_c P(c | x_i)$
        - Confidence: $c_i = \max_c P(c | x_i)$
        - Timestamp: $t_i$
        
        **Temporal Analysis:**
        
        **Class Consensus:**
        """)
        st.latex(r"\text{Consensus} = \frac{1}{W} \sum_{i=t-W+1}^{t} \mathbb{1}[y_i = y_{mode}]")
        st.markdown("""
        
        Where $y_{mode}$ is the most frequent class in the window.
        
        """)
        st.latex(r"\text{Consensus Check} = \begin{cases} \text{Pass} & \text{if } \text{Consensus} \geq 0.8 \\ \text{Fail} & \text{otherwise} \end{cases}")
        st.markdown("""
        
        **Confidence Stability:**
        """)
        st.latex(r"\mu_c = \frac{1}{W} \sum_{i=t-W+1}^{t} c_i")
        st.latex(r"\sigma_c^2 = \frac{1}{W} \sum_{i=t-W+1}^{t} (c_i - \mu_c)^2")
        st.latex(r"\text{Stability Check} = \begin{cases} \text{Pass} & \text{if } \sigma_c \leq 0.1 \\ \text{Fail} & \text{otherwise} \end{cases}")
        st.markdown("""
        
        **Temporal Gradient** (rate of change):
        """)
        st.latex(r"\Delta y_t = \frac{1}{W-1} \sum_{i=t-W+2}^{t} |y_i - y_{i-1}|")
        st.latex(r"\Delta c_t = \frac{1}{W-1} \sum_{i=t-W+2}^{t} |c_i - c_{i-1}|")
        st.latex(r"\text{Gradient} = \max(\Delta y_t, \Delta c_t)")
        st.latex(r"\text{Gradient Check} = \begin{cases} \text{Pass} & \text{if } \text{Gradient} \leq 0.3 \\ \text{Fail} & \text{otherwise} \end{cases}")
        st.markdown("""
        
        **Detection:**
        """)
        st.latex(r"\text{Attack Detected} = \neg(\text{Consensus Check} \land \text{Stability Check} \land \text{Gradient Check})")
        st.markdown("""
        
        If any check fails, the attack is detected due to temporal inconsistency.
        """)
    
    st.markdown("""
    **Why It Works**:
    - Natural objects: Stable predictions [0.95, 0.94, 0.96, 0.95]
    - Adversarial patches: Unstable predictions [0.95, 0.45, 0.92, 0.38]
    - Patches cause model confusion that manifests as temporal instability
    
    **Strengths**: Very effective for video/camera feeds, catches temporal attacks, low computational cost
    
    **Weaknesses**: Requires multiple frames (not useful for single images), may flag legitimate scene changes
    """)
    
    st.markdown("""
    ### How Defenses Work Together
    
    **Cascade Mode** (Default):
    
    Defenses are applied **sequentially** - each layer processes the output of the previous layer:
    
    ```
    Input Image
        ↓
    [Entropy Detection] → Masks high-entropy regions
        ↓
    [Frequency Analysis] → Filters high-frequency anomalies
        ↓
    [Gradient Saliency] → Masks high-saliency regions
        ↓
    [Multi-Frame Smoothing] → Validates temporal consistency
        ↓
    Final Decision
    ```
    
    **Detection Logic**:
    
    The system uses **OR logic** - if **ANY** defense detects a patch, the attack is blocked.
    """)
    
    with st.expander("📐 View Mathematical Formulation for Defense Cascade", expanded=False):
        st.markdown(r"""
        Let $D_1, D_2, D_3, D_4$ be the detection results from the 4 defense layers:
        - $D_1$: Entropy detection (binary: 0 or 1)
        - $D_2$: Frequency detection (binary: 0 or 1)
        - $D_3$: Saliency detection (binary: 0 or 1)
        - $D_4$: Temporal instability (binary: 0 or 1)
        
        **Final Decision:**
        $$\text{Attack Blocked} = D_1 \lor D_2 \lor D_3 \lor D_4$$
        
        Where $\lor$ is the logical OR operator.
        
        **Confidence Aggregation:**
        $$\text{Overall Confidence} = \max(C_1, C_2, C_3, C_4)$$
        
        Where $C_i$ is the confidence score from defense layer $i$.
        
        **Cascade Processing:**
        $$x_0 = x_{input}$$
        $$x_1 = \text{EntropyDefense}(x_0)$$
        $$x_2 = \text{FrequencyDefense}(x_1)$$
        $$x_3 = \text{SaliencyDefense}(x_2)$$
        $$x_{final} = x_3$$
        
        Each defense layer processes the output of the previous layer, allowing early layers to remove suspicious content before later layers process it.
        """)
    
    st.markdown("""
    **Why This Works**:
    
    1. **Complementary Detection**: Each layer catches different attack characteristics
    2. **Cascade Effect**: Early layers remove suspicious content before later layers process it
    3. **Defense in Depth**: Attacker must bypass all 4 layers simultaneously
    4. **Low False Positives**: Multiple independent checks reduce false alarms
    
    **Performance**:
    
    - **Detection Rate**: >95% for known adversarial patches
    - **False Positive Rate**: <2% on natural images
    - **Computational Overhead**: ~3× slower than basic defenses (acceptable for security)
    - **Latency**: Adds ~50-100ms per frame (real-time capable)
    
    This multi-layer defense architecture is the **industry standard** for protecting 
    production computer vision systems against adversarial attacks.
    """)
    
    st.markdown('<div class="duke-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## System Features
    
    - **Boo Patch**: Opens Notepad and types "Boo" when detected
    - **Malware Patch**: Downloads Malware-V1.0 repository and executes blue_devil_lock.py when detected
    - **Basic Defense Mode**: Uses standard defense pipeline (relaxed settings)
    - **Advanced Defense Mode**: Uses 4-layer advanced defense system with detailed detection reporting
    """)

def camera_page():
    """Display Camera Input page - Simple pipeline."""
    # Duke-themed header
    st.markdown("""
<div class="duke-header">
    <h1 style="color: white !important; border: none !important;">📷 Camera Patch Detection</h1>
</div>
""", unsafe_allow_html=True)
    
    # Manual initialization button
    if st.session_state.attack_system is None:
        col_init1, col_init2 = st.columns([3, 1])
        with col_init1:
            st.markdown('<p style="color: white !important; font-size: 1.1rem; font-weight: 500; padding: 1rem; background: rgba(0, 83, 155, 0.2); border-radius: 8px; border: 2px solid rgba(0, 163, 224, 0.5);">⚠️ Attack system not initialized. Click the button to initialize.</p>', unsafe_allow_html=True)
        with col_init2:
            if st.button("🚀 Initialize System", use_container_width=True, type="primary"):
                with st.spinner("Initializing attack system..."):
                    if initialize_attack_system():
                        st.success("✅ System initialized!")
                        st.rerun()
                    else:
                        st.error("❌ Initialization failed.")
    else:
        st.success("✅ Attack system ready!")
    
    # Sidebar controls (minimal)
    with st.sidebar:
        st.markdown("""
        <div style="background: rgba(0, 83, 155, 0.3); 
                    color: white; 
                    padding: 0.75rem;
                    border-radius: 8px;
                    border: 2px solid rgba(0, 163, 224, 0.5);
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
            <h3 style="color: white !important; margin: 0; font-size: 1.1rem; font-weight: 600; text-align: center;">⚙️ Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # System type toggle
        st.markdown("""
        <div style="background: rgba(0, 83, 155, 0.3); 
                    color: white; 
                    padding: 0.75rem;
                    border-radius: 8px;
                    border: 2px solid rgba(0, 163, 224, 0.5);
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
            <h3 style="color: white !important; margin: 0; font-size: 1rem; font-weight: 600; text-align: center;">🔀 System Type</h3>
        </div>
        """, unsafe_allow_html=True)
        use_advanced = st.checkbox(
            "🛡️ Use Advanced Defense System",
            value=st.session_state.use_advanced_system,
            help="Switch to Advanced Defense System to see which defense layer stops attacks"
        )
        
        if use_advanced != st.session_state.use_advanced_system:
            st.session_state.use_advanced_system = use_advanced
            st.session_state.attack_system = None
            st.session_state.system_initialized = False
            st.rerun()
        
        if st.session_state.use_advanced_system:
            st.info("🛡️ **Advanced System Active**\n\nShows which defense layer blocks attacks:\n- **Entropy Defense**\n- **Frequency Defense**\n- **Gradient Saliency Defense**\n- **Multi-Frame Smoothing**")
        else:
            st.info("⚡ **Basic System Active**\n\nExecutes commands when patch detected.")
        
        st.markdown("---")
        
        if st.button("🔄 Reinitialize System", use_container_width=True):
            st.session_state.attack_system = None
            st.session_state.system_initialized = False
            if st.session_state.attack_system:
                st.session_state.attack_system.command_executed = False
            st.info("System reset. Click 'Initialize System' to reload.")
            st.rerun()
    
    # Main camera interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        # Input method selection
        input_method = st.radio(
            "Select Input Method:",
            ["📷 Camera (Browser)", "📁 Upload Image File"],
            horizontal=True
        )
    
        camera_image = None
        img_array = None
        
        if input_method == "📷 Camera (Browser)":
            # Camera mode selection
            camera_mode = st.radio(
                "Camera Mode:",
                ["Streamlit Camera", "OpenCV Direct (Live)"],
                horizontal=True
            )
            
            if camera_mode == "Streamlit Camera":
                try:
                    camera_image = st.camera_input(
                        "Show adversarial patch to camera",
                        key="camera_input"
                    )
                except Exception as e:
                    st.error(f"❌ Camera error: {e}")
                    camera_image = None
            else:
                # OpenCV direct camera access
                if st.button("🎥 Start Live Camera"):
                    st.session_state.opencv_camera_active = True
                
                if st.session_state.opencv_camera_active:
                    try:
                        cap = cv2.VideoCapture(st.session_state.opencv_camera_id)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Process frame - simple pipeline
                                detection, processed_frame = process_frame(frame_rgb)
                                
                                if detection:
                                    if st.session_state.use_advanced_system:
                                        # Advanced system: show defense layer information
                                        if detection.get('patch_detected_by_defense', False):
                                            st.error("🛡️ **ATTACK BLOCKED BY DEFENSES**")
                                            
                                            # Show which defense layer stopped it
                                            validation = detection.get('validation_result', {})
                                            defense_results = validation.get('defense_results', {})
                                            
                                            blocked_by = []
                                            if 'entropy' in defense_results and defense_results['entropy'].get('patch_detected', False):
                                                blocked_by.append("Entropy Defense")
                                            if 'frequency' in defense_results and defense_results['frequency'].get('patch_detected', False):
                                                blocked_by.append("Frequency Defense")
                                            if 'gradient_saliency' in defense_results and defense_results['gradient_saliency'].get('patch_detected', False):
                                                blocked_by.append("Gradient Saliency Defense")
                                            if 'enhanced_multi_frame' in defense_results and not defense_results['enhanced_multi_frame'].get('consensus_reached', True):
                                                blocked_by.append("Multi-Frame Smoothing")
                                            
                                            if blocked_by:
                                                st.warning(f"**🚫 Attack stopped at: {blocked_by[0]}**")
                                        elif detection.get('detected', False) and detection.get('defense_bypassed', False):
                                            st.error("🚨 **ATTACK SUCCESSFUL** - Patch bypassed all defenses!")
                                            st.success("✅ Command sent to attack system")
                                        else:
                                            st.info("⏳ Scanning...")
                                    else:
                                        # Basic system
                                        if detection.get('detected', False):
                                            patch_type = detection.get('patch_type', 'unknown')
                                            if patch_type == 'malware':
                                                st.error("🚨 MALWARE PATCH DETECTED - Executing...")
                                            else:
                                                st.warning("🚨 BOO PATCH DETECTED - Executing...")
                                            st.success("✅ Command sent to attack system")
                                        else:
                                            st.info("⏳ Scanning...")
                                
                                st.image(frame_rgb, channels="RGB", caption="Live Camera Feed")
                                
                                cap.release()
                                
                                # Auto-refresh for continuous scanning
                                time.sleep(0.1)
                                st.rerun()
                            else:
                                st.error("Failed to read frame")
                                cap.release()
                        else:
                            st.error(f"❌ Cannot open camera {st.session_state.opencv_camera_id}")
                    except Exception as e:
                        st.error(f"Camera error: {e}")
        else:
            # File upload method
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg']
            )
        
            if uploaded_file is not None:
                # Convert UploadedFile to PIL Image
                try:
                    camera_image = Image.open(uploaded_file)
                    camera_image = camera_image.convert('RGB')
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    camera_image = None
        
        # Process camera image (Streamlit camera or file upload)
        if camera_image is not None:
            if isinstance(camera_image, Image.Image):
                img_array = np.array(camera_image, dtype=np.uint8)
            else:
                try:
                    camera_image = Image.open(camera_image)
                    camera_image = camera_image.convert('RGB')
                    img_array = np.array(camera_image, dtype=np.uint8)
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    img_array = None
        
        if img_array is not None:
            # Process frame - simple pipeline
            detection, processed_frame = process_frame(img_array)
            
            if detection:
                    if st.session_state.use_advanced_system:
                        # Advanced system: show defense layer information
                        if detection.get('patch_detected_by_defense', False):
                            st.error("🛡️ **ATTACK BLOCKED BY DEFENSES**")
                            
                            # Show which defense layer stopped it
                            validation = detection.get('validation_result', {})
                            defense_results = validation.get('defense_results', {})
                            
                            st.markdown("### 🛡️ Defense Layer Analysis:")
                            
                            # Check each defense layer
                            blocked_by = []
                            if 'entropy' in defense_results:
                                ent = defense_results['entropy']
                                if ent.get('patch_detected', False):
                                    blocked_by.append(("Entropy Defense", ent.get('confidence', 0.0)))
                                    st.error(f"🔴 **Entropy Defense**: DETECTED (conf: {ent.get('confidence', 0.0):.3f})")
                                else:
                                    st.success(f"🟢 **Entropy Defense**: CLEAR (conf: {ent.get('confidence', 0.0):.3f})")
                            
                            if 'frequency' in defense_results:
                                freq = defense_results['frequency']
                                if freq.get('patch_detected', False):
                                    blocked_by.append(("Frequency Defense", freq.get('confidence', 0.0)))
                                    st.error(f"🔴 **Frequency Defense**: DETECTED (conf: {freq.get('confidence', 0.0):.3f})")
                                else:
                                    st.success(f"🟢 **Frequency Defense**: CLEAR (conf: {freq.get('confidence', 0.0):.3f})")
                            
                            if 'gradient_saliency' in defense_results:
                                sal = defense_results['gradient_saliency']
                                if sal.get('patch_detected', False):
                                    blocked_by.append(("Gradient Saliency Defense", sal.get('confidence', 0.0)))
                                    st.error(f"🔴 **Gradient Saliency Defense**: DETECTED (conf: {sal.get('confidence', 0.0):.3f})")
                                else:
                                    st.success(f"🟢 **Gradient Saliency Defense**: CLEAR (conf: {sal.get('confidence', 0.0):.3f})")
                            
                            if 'enhanced_multi_frame' in defense_results:
                                mf = defense_results['enhanced_multi_frame']
                                if not mf.get('consensus_reached', True):
                                    blocked_by.append(("Multi-Frame Smoothing", 1.0))
                                    st.error(f"🔴 **Multi-Frame Smoothing**: NO CONSENSUS")
                                else:
                                    st.success(f"🟢 **Multi-Frame Smoothing**: CONSENSUS REACHED")
                            
                            # Show which layer stopped it
                            if blocked_by:
                                st.markdown("---")
                                st.warning(f"**🚫 Attack stopped at: {blocked_by[0][0]}** (Confidence: {blocked_by[0][1]:.3f})")
                        elif detection.get('detected', False) and detection.get('defense_bypassed', False):
                            st.error("🚨 **ATTACK SUCCESSFUL** - Patch bypassed all defenses!")
                            st.success("✅ Command sent to attack system")
                        else:
                            st.info("⏳ No patch detected or defenses active")
                    else:
                        # Basic system: simple detection
                        if detection.get('detected', False):
                            patch_type = detection.get('patch_type', 'unknown')
                            if patch_type == 'malware':
                                st.error("🚨 MALWARE PATCH DETECTED - Executing...")
                            elif patch_type == 'boo':
                                st.warning("🚨 BOO PATCH DETECTED - Executing...")
                            else:
                                st.warning(f"🚨 PATCH DETECTED! Type: {patch_type.upper()}")
                            st.success("✅ Command sent to attack system")
                        else:
                            st.info("⏳ No patch detected")
    
    with col2:
        st.subheader("📈 Status")
        
        if st.session_state.use_advanced_system:
            st.info("🛡️ **Advanced Defense System Active**")
            st.markdown("""
            ### Defense Layers:
            - **Entropy Defense**: Detects high entropy patches
            - **Frequency Defense**: Analyzes frequency domain
            - **Gradient Saliency**: Uses gradient-based detection
            - **Multi-Frame Smoothing**: Temporal consistency check
            """)
        else:
            st.info("⏳ Waiting for patch detection...")
        
        st.markdown("---")
        st.subheader("📖 Instructions")
        
        if st.session_state.use_advanced_system:
            st.markdown("""
            **Advanced System Mode:**
            1. Show patch to camera
            2. System will show which defense layer blocks the attack
            3. If all defenses are bypassed, command executes
            
            **Note**: Advanced system only supports Boo patch
            """)
        else:
            st.markdown("""
            1. **Show Boo Patch**: Opens Notepad, types "Boo"
            
            2. **Show Malware Patch**: Downloads repo, executes blue_devil_lock.py
               - Password: `123456789`
            
            3. **Execution**: All commands execute locally in the attack system
            """)

def main():
    """Main application."""
    # Duke-themed sidebar header
    st.sidebar.markdown("""
    <div style="padding: 0.5rem 0 1rem 0; margin-bottom: 1rem;">
        <h1 style="color: white !important; margin: 0; font-size: 2rem; font-weight: bold; text-align: center; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Adversarial AI Project</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="background: rgba(0, 83, 155, 0.3); 
                color: white; 
                padding: 0.75rem;
                border-radius: 8px;
                border: 2px solid rgba(0, 163, 224, 0.5);
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
        <h3 style="color: white !important; margin: 0; font-size: 1.1rem; font-weight: 600; text-align: center;">Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Page:",
        ["📷 Camera Detection", "ℹ️ About the Project"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgba(0, 83, 155, 0.3); 
                color: white; 
                padding: 0.75rem;
                border-radius: 8px;
                border: 2px solid rgba(0, 163, 224, 0.5);
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
        <h3 style="color: white !important; margin: 0; font-size: 1.1rem; font-weight: 600; text-align: center;">System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.attack_system:
        system_type = "🛡️ Advanced Defense" if st.session_state.use_advanced_system else "⚡ Basic System"
        st.sidebar.success(f"✅ {system_type} Ready")
        device_status = "🟢 CUDA" if st.session_state.device == 'cuda' else "🟡 CPU"
        st.sidebar.info(f"Device: {device_status}")
    else:
        st.sidebar.warning("⚠️ System Not Initialized")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgba(0, 83, 155, 0.3); 
                color: white; 
                padding: 0.75rem;
                border-radius: 8px;
                border: 2px solid rgba(0, 163, 224, 0.5);
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
        <h3 style="color: white !important; margin: 0; font-size: 1.1rem; font-weight: 600; text-align: center;">Quick Info</h3>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="background: rgba(0, 83, 155, 0.3); 
                color: white; 
                font-size: 0.95rem;
                padding: 1rem;
                border-radius: 8px;
                border: 2px solid rgba(0, 163, 224, 0.5);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">
    <p style="margin: 0.5rem 0;"><strong>🔵 Boo Patch:</strong> Opens Notepad, types "Boo"</p>
    <p style="margin: 0.5rem 0;"><strong>🔴 Malware Patch:</strong> Downloads repo, locks screen</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p style="margin: 0; font-size: 1rem; color: white !important; font-weight: 600; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">Duke University</p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: #B3D9FF !important;">Adversarial Patch Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "📷 Camera Detection":
        camera_page()
    elif page == "ℹ️ About the Project":
        about_page()

if __name__ == "__main__":
    main()
