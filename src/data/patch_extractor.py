import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
try:
    import openslide
except ImportError:
    raise ImportError("openslide-python is required. Please install it and ensure OpenSlide binaries are in your PATH.")

class PatchExtractor:
    """
    Extracts tissue-only patches from Whole-Slide Images (WSIs) using OpenSlide and OpenCV.
    """
    def __init__(self, 
                 patch_size: int = 256, 
                 magnification_target: int = 20, 
                 background_threshold: int = 210,
                 tissue_ratio: float = 0.5):
        """
        Args:
            patch_size: Size of the square patch (e.g., 256x256).
            magnification_target: Target magnification (e.g., 20x).
            background_threshold: Pixel intensity above which is considered background (white).
            tissue_ratio: Minimum fraction of tissue required in a patch to keep it.
        """
        self.patch_size = patch_size
        self.magnification_target = magnification_target
        self.bg_thresh = background_threshold
        self.tissue_ratio = tissue_ratio

    def get_target_level(self, slide: openslide.OpenSlide) -> int:
        """Determines the best OpenSlide level for the target magnification."""
        # TCGA images are typically 40x at level 0.
        # If we want 20x, we look for a level with downsample roughly ~2.
        # For simplicity in this prototype, we'll try to find the level closest to 20x,
        # assuming level 0 is 40x. If properties missing, default to level 1.
        
        try:
            base_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40.0))
        except ValueError:
            base_mag = 40.0
            
        target_downsample = base_mag / self.magnification_target
        
        # Find the level with the downsample closest to our target
        best_level = slide.get_best_level_for_downsample(target_downsample)
        return best_level

    def is_tissue(self, patch: np.ndarray) -> bool:
        """Checks if a patch contains enough tissue based on thresholding."""
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Pixels darker than the background threshold are considered tissue
        tissue_pixels = np.sum(gray < self.bg_thresh)
        total_pixels = self.patch_size * self.patch_size
        
        return (tissue_pixels / total_pixels) >= self.tissue_ratio

    def extract_patches(self, slide_path: Path, max_patches: int = 500) -> List[np.ndarray]:
        """
        Extracts tissue patches from the WSI.
        Returns a list of valid RGB patches (numpy arrays).
        """
        slide = openslide.OpenSlide(str(slide_path))
        level = self.get_target_level(slide)
        
        dimensions = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        # Calculate step size at level 0
        step_size_lvl0 = int(self.patch_size * downsample)
        width_lvl0, height_lvl0 = slide.dimensions
        
        valid_patches = []
        
        # Grid search for patches
        # For efficiency on gigapixel images, we step by the patch size
        for y in range(0, height_lvl0 - step_size_lvl0, step_size_lvl0):
            for x in range(0, width_lvl0 - step_size_lvl0, step_size_lvl0):
                
                if len(valid_patches) >= max_patches:
                    break
                    
                # Read region (must provide coordinates at level 0, but size at target level)
                try:
                    region = slide.read_region((x, y), level, (self.patch_size, self.patch_size))
                except Exception:
                    continue
                    
                # Convert RGBA PIL image to RGB numpy array
                region_rgb = np.array(region.convert('RGB'))
                
                # Check if it's mostly tissue
                if self.is_tissue(region_rgb):
                    valid_patches.append(region_rgb)
                    
            if len(valid_patches) >= max_patches:
                break
                
        slide.close()
        return valid_patches
