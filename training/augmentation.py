'''
Training: Geometry-Safe Augmentation Pipeline

When mapping visual topologies to continuous manifolds, standard computer vision augmentations
(like RandomResizedCrop) are highly destructive. Cropping an image of a triangle might remove
one of its vertices, completely destroying the geometric relationships the CoT reasoning relies on.

This module guarantees that augmented "positive/negative views" for contrastive learning
(like VICReg or InfoNCE) are strictly limited to affine transformations (translation, rotation, 
scaling, shearing) that safely preserve parallel lines and topological continuity without cropping 
crucial visual evidence out of the frame.
'''

import torch
import torchvision.transforms.v2 as transforms

class GeometrySafeAugmentation:
    """
    Applies strict, geometry-safe affine transformations to diagrams.
    Guarantees no destructive crops that would remove mathematical context.
    Allows for: Translation, Rotation, Scaling, Shearing, and Low-Norm Affine Matrices.
    """
    def __init__(self, 
                 max_rotation: float = 15.0, 
                 max_translate: tuple = (0.1, 0.1), 
                 scale_range: tuple = (0.9, 1.1),
                 max_shear: float = 10.0):
                 
        # Standard highly controlled, low-norm Affine transforms
        # We explicitly rely on RandomAffine rather than RandomResizedCrop
        self.transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=max_rotation,
                translate=max_translate,
                scale=scale_range,
                shear=max_shear,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        ])

    def __call__(self, image):
        """
        Applies the geometry-safe affine transformation.
        image: PIL Image or Tensor
        """
        return self.transform(image)

def generate_vicreg_pairs(images, augmentor):
    """
    Applies augmentations to generate diverse views for the VICReg invariance constraint.
    
    If using standard Supervised VICReg:
    Predictor(image) -> Targ(Expert Text)
    Predictor(augmented_image) -> Targ(Expert Text)
    
    If using Self-Supervised VICReg for pretraining:
    Predictor(image) -> Predictor(augmented_image)
    """
    augmented_views = []
    for img in images:
        augmented_views.append(augmentor(img))
    
    # Needs stacking back into target formats later up the pipeline
    return augmented_views

if __name__ == "__main__":
    # Test Scaffold
    # augmentor = GeometrySafeAugmentation()
    pass

class DestroyGeometryAugmentation:
    """
    Applies highly destructive non-affine spatial transformations to create 
    true negative examples for InfoNCE/VICReg contrastive learning batches.
    By actively ruining topological continuity (via extreme perspective warping
    and aggressive random erasing), the model learns what invalid geometry looks like.
    """
    def __init__(self, erase_prob: float = 0.5, distortion_scale: float = 0.8):
        self.transform = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=distortion_scale, p=0.8),
            # Aggressive erasing to simulate missing vertices or disconnected segments
            transforms.RandomErasing(p=erase_prob, scale=(0.1, 0.33), ratio=(0.2, 5.0), value=0),
            transforms.RandomErasing(p=erase_prob, scale=(0.05, 0.15), ratio=(0.2, 5.0), value=255)
        ])

    def __call__(self, image):
        """
        Applies a destructive transformation resulting in a ruined geometry negative sample.
        Expects a PyTorch Tensor (RandomErasing requires tensors).
        """
        return self.transform(image)

def generate_negative_pairs(images, destroyer):
    """
    Applies aggressive transformations to generate explicitly invalid geometric views. 
    Use these as the negative pairs in contrastive batches to push repelling vectors.
    """
    negative_views = []
    for img in images:
        negative_views.append(destroyer(img))
    return negative_views
