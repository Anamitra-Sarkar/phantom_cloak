"""
VFX Utilities for PHANTOM-CLOAK
Real-Time Optical Camouflage System

This module contains mathematical functions for distortion and shimmer effects.
"""

import cv2
import numpy as np
from collections import OrderedDict

# Cache for meshgrid coordinates to avoid recreation
# Using OrderedDict with max size to prevent unbounded memory growth
# Note: This cache is not thread-safe. If using in multi-threaded context, add synchronization.
_MESHGRID_CACHE_MAX_SIZE = 10  # Reasonable limit for different resolutions
_meshgrid_cache = OrderedDict()

# Pre-computed constants for performance
_INV_255 = 1.0 / 255.0  # Avoid repeated division calculations

# HUD Scanline effect constants
HUD_SCANLINE_STEP = 4       # Step between scanlines
HUD_SCANLINE_MODULO = 8     # Modulo for alternating scanlines
HUD_SCANLINE_TOP = 5        # Pixels from bottom for scanline top
HUD_SCANLINE_BOTTOM = 3     # Pixels from bottom for scanline bottom
HUD_SCANLINE_WIDTH = 2      # Width of each scanline segment


def detect_edges(mask: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges in the segmentation mask using Canny edge detection.
    
    Args:
        mask: Binary segmentation mask (0-255)
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
    
    Returns:
        Edge mask with detected edges
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    edges = cv2.Canny(mask_uint8, low_threshold, high_threshold)
    return edges


def create_displacement_maps(height: int, width: int, edges: np.ndarray, 
                              displacement_strength: float = 10.0,
                              time_offset: float = 0.0,
                              dilated_edges: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Create displacement maps for light bending effect.
    
    Args:
        height: Frame height
        width: Frame width
        edges: Edge mask from detect_edges
        displacement_strength: Strength of the displacement effect
        time_offset: Time offset for animated shimmer effect
        dilated_edges: Pre-dilated edges (optional, to avoid redundant dilation)
    
    Returns:
        Tuple of (map_x, map_y) for cv2.remap
    """
    # Use cached meshgrid coordinates to avoid recreation
    cache_key = (height, width)
    if cache_key not in _meshgrid_cache:
        x_coords, y_coords = np.meshgrid(np.arange(width, dtype=np.float32), 
                                         np.arange(height, dtype=np.float32))
        _meshgrid_cache[cache_key] = (x_coords, y_coords)
        
        # Implement LRU-style cache eviction if cache grows too large
        if len(_meshgrid_cache) > _MESHGRID_CACHE_MAX_SIZE:
            _meshgrid_cache.popitem(last=False)  # Remove oldest item
    else:
        # Move to end to mark as recently used
        _meshgrid_cache.move_to_end(cache_key)
        x_coords, y_coords = _meshgrid_cache[cache_key]
    
    # Use pre-dilated edges if provided to avoid redundant dilation
    if dilated_edges is None:
        dilated_edges = cv2.dilate(edges, np.ones((15, 15), np.uint8), iterations=2)
    
    edge_region = dilated_edges * _INV_255
    
    # Pre-compute displacement with efficient operations
    noise_x = np.sin(y_coords * 0.1 + time_offset) * displacement_strength
    noise_y = np.cos(x_coords * 0.1 + time_offset) * displacement_strength
    
    # In-place operations where possible
    map_x = np.clip(x_coords + noise_x * edge_region, 0, width - 1)
    map_y = np.clip(y_coords + noise_y * edge_region, 0, height - 1)
    
    return map_x, map_y


def apply_predator_shimmer(frame: np.ndarray, background: np.ndarray, 
                           mask: np.ndarray, edges: np.ndarray,
                           time_offset: float = 0.0,
                           shimmer_intensity: float = 0.3,
                           refraction_index: float = 1.4) -> np.ndarray:
    """
    Apply the Predator-style shimmer distortion effect.
    
    This simulates light bending around the cloaked figure by:
    1. Applying displacement to the background at edge regions
    2. Blending with slight transparency for the shimmer effect
    3. Adding chromatic aberration for realism
    
    Args:
        frame: Current camera frame
        background: Captured background plate
        mask: Segmentation mask (normalized 0-1)
        edges: Detected edges from the mask
        time_offset: Time for animated effects
        shimmer_intensity: Intensity of the shimmer effect (0-1)
        refraction_index: Simulated refraction index for distortion strength
    
    Returns:
        Processed frame with predator shimmer effect
    """
    height, width = frame.shape[:2]
    
    displacement_strength = (refraction_index - 1.0) * 25.0
    
    # Pre-dilate edges once for both displacement and shimmer
    dilated_edges_large = cv2.dilate(edges, np.ones((15, 15), np.uint8), iterations=2)
    
    map_x, map_y = create_displacement_maps(
        height, width, edges, 
        displacement_strength=displacement_strength,
        time_offset=time_offset,
        dilated_edges=dilated_edges_large
    )
    
    distorted_bg = cv2.remap(background, map_x, map_y, 
                              cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REFLECT)
    
    chromatic_offset = int(displacement_strength * 0.3)
    if chromatic_offset > 0:
        b, g, r = cv2.split(distorted_bg)
        rows, cols = b.shape
        
        shift_matrix_r = np.float32([[1, 0, chromatic_offset], [0, 1, 0]])
        shift_matrix_b = np.float32([[1, 0, -chromatic_offset], [0, 1, 0]])
        
        r = cv2.warpAffine(r, shift_matrix_r, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b = cv2.warpAffine(b, shift_matrix_b, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        distorted_bg = cv2.merge([b, g, r])
    
    # Optimize mask expansion using dstack instead of stack
    mask_3ch = np.dstack([mask, mask, mask])
    
    # Use smaller dilation for edge shimmer effect
    dilated_edges_small = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)
    edge_mask = dilated_edges_small * _INV_255
    edge_mask_3ch = np.dstack([edge_mask, edge_mask, edge_mask])
    
    # Vectorized blending operations
    cloak_result = distorted_bg * mask_3ch + frame * (1.0 - mask_3ch)
    
    shimmer_noise = (np.sin(time_offset * 5.0) + 1.0) * 0.5 * shimmer_intensity
    shimmer_blend = cloak_result * (1.0 - edge_mask_3ch * shimmer_noise) + \
                    frame * (edge_mask_3ch * shimmer_noise)
    
    return shimmer_blend.astype(np.uint8)


def apply_absolute_invisibility(frame: np.ndarray, background: np.ndarray,
                                 mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply absolute invisibility effect - clean background replacement.
    
    Args:
        frame: Current camera frame
        background: Captured background plate
        mask: Segmentation mask (normalized 0-1)
        threshold: Threshold for mask application
    
    Returns:
        Frame with person replaced by background
    """
    binary_mask = (mask > threshold).astype(np.float32)
    
    blurred_mask = cv2.GaussianBlur(binary_mask, (21, 21), 0)
    # Use dstack for better performance than stack
    mask_3ch = np.dstack([blurred_mask, blurred_mask, blurred_mask])
    
    # Vectorized blending
    result = background * mask_3ch + frame * (1.0 - mask_3ch)
    
    return result.astype(np.uint8)


def refine_mask(mask: np.ndarray, blur_size: int = 15) -> np.ndarray:
    """
    Refine the segmentation mask for seamless blending.
    
    Args:
        mask: Raw segmentation mask
        blur_size: Size of Gaussian blur kernel (must be odd)
    
    Returns:
        Refined mask with soft edges
    """
    # Ensure blur_size is odd
    if blur_size % 2 == 0:
        blur_size += 1
    
    refined = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    return refined


def create_hud_overlay(frame: np.ndarray, mode: str, refraction_index: float,
                       fps: float, calibrating: bool = False,
                       countdown: int = 0) -> np.ndarray:
    """
    Create the military prototype HUD overlay.
    
    Args:
        frame: Current frame to overlay HUD on
        mode: Current mode ("ABSOLUTE" or "PREDATOR")
        refraction_index: Current refraction index value
        fps: Current FPS
        calibrating: Whether calibration is in progress
        countdown: Countdown timer for calibration
    
    Returns:
        Frame with HUD overlay
    """
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    green = (0, 255, 0)
    dark_green = (0, 180, 0)
    red = (0, 0, 255)
    
    corner_size = 50
    corner_thickness = 2
    cv2.line(overlay, (10, 10), (10 + corner_size, 10), green, corner_thickness)
    cv2.line(overlay, (10, 10), (10, 10 + corner_size), green, corner_thickness)
    cv2.line(overlay, (width - 10, 10), (width - 10 - corner_size, 10), green, corner_thickness)
    cv2.line(overlay, (width - 10, 10), (width - 10, 10 + corner_size), green, corner_thickness)
    cv2.line(overlay, (10, height - 10), (10 + corner_size, height - 10), green, corner_thickness)
    cv2.line(overlay, (10, height - 10), (10, height - 10 - corner_size), green, corner_thickness)
    cv2.line(overlay, (width - 10, height - 10), (width - 10 - corner_size, height - 10), green, corner_thickness)
    cv2.line(overlay, (width - 10, height - 10), (width - 10, height - 10 - corner_size), green, corner_thickness)
    
    if calibrating:
        cal_text = f"CLEAR THE FRAME. CAPTURING BACKGROUND IN {countdown}..."
        text_size = cv2.getTextSize(cal_text, font, 0.8, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        cv2.putText(overlay, cal_text, (text_x, text_y), font, 0.8, red, 2)
    else:
        cv2.putText(overlay, "PHANTOM-CLOAK v1.0", (20, 35), font, font_scale, green, thickness)
        cv2.putText(overlay, "ACTIVE CAMO: ENABLED", (20, 60), font, font_scale - 0.1, green, 1)
        
        mode_text = f"MODE: {mode}"
        cv2.putText(overlay, mode_text, (20, 85), font, font_scale - 0.1, dark_green, 1)
        
        refraction_text = f"REFRACTION INDEX: {refraction_index:.1f}"
        cv2.putText(overlay, refraction_text, (20, 110), font, font_scale - 0.1, dark_green, 1)
        
        fps_text = f"FPS: {fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, font, font_scale - 0.1, 1)[0]
        cv2.putText(overlay, fps_text, (width - fps_size[0] - 20, 35), font, font_scale - 0.1, green, 1)
        
        controls_y = height - 30
        cv2.putText(overlay, "[C] RECALIBRATE", (20, controls_y), font, 0.4, dark_green, 1)
        cv2.putText(overlay, "[M] SWITCH MODE", (180, controls_y), font, 0.4, dark_green, 1)
        cv2.putText(overlay, "[Q] QUIT", (340, controls_y), font, 0.4, dark_green, 1)
    
    # Draw HUD scanline effect at bottom of frame
    for i in range(0, width, HUD_SCANLINE_STEP):
        if i % HUD_SCANLINE_MODULO == 0:
            overlay[height - HUD_SCANLINE_TOP:height - HUD_SCANLINE_BOTTOM, 
                    i:i + HUD_SCANLINE_WIDTH] = green
    
    return overlay
