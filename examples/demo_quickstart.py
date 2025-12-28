#!/usr/bin/env python
"""
AugmentLens Quickstart Demo

This script demonstrates the full AugmentLens workflow:
1. Create synthetic test data (no external files needed)
2. Wrap an Albumentations pipeline with PipelineHook
3. Run augmentation and capture before/after
4. Render the comparison with MatplotlibRenderer

Run with: python examples/demo_quickstart.py
"""

import numpy as np
import albumentations as A

from augmentlens import PipelineHook, MatplotlibRenderer


def create_dummy_data() -> tuple[np.ndarray, list, list]:
    """
    Generate synthetic image and bounding box for testing.
    
    Creates a 512x512 image with a colorful circle pattern and
    a bounding box around the central region.
    
    Returns
    -------
    tuple
        (image, bboxes, bbox_labels)
    """
    size = 512
    center = size // 2
    
    # Create coordinate grids for vectorized operations
    y_coords, x_coords = np.ogrid[:size, :size]
    
    # Build gradient background using broadcasting (no loops!)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = (255 * np.arange(size) / size).astype(np.uint8)  # Red: horizontal
    image[:, :, 1] = (255 * np.arange(size) / size).astype(np.uint8)[:, np.newaxis]  # Green: vertical
    image[:, :, 2] = 128  # Blue: constant
    
    # Draw white outer circle using boolean mask
    outer_radius = 100
    outer_circle_mask = ((x_coords - center) ** 2 + (y_coords - center) ** 2) <= outer_radius ** 2
    image[outer_circle_mask] = [255, 255, 255]
    
    # Draw orange inner circle
    inner_radius = 50
    inner_circle_mask = ((x_coords - center) ** 2 + (y_coords - center) ** 2) <= inner_radius ** 2
    image[inner_circle_mask] = [255, 100, 50]
    
    # Bounding box around the central region
    bboxes = [(center - 120, center - 120, center + 120, center + 120)]
    bbox_labels = ["target"]
    
    return image, bboxes, bbox_labels


def main():
    """Run the full demo pipeline."""
    print("=" * 60)
    print("AugmentLens Quickstart Demo")
    print("=" * 60)
    
    # Step 1: Create synthetic data
    print("\n[Step 1] Creating synthetic test data...")
    image, bboxes, bbox_labels = create_dummy_data()
    print(f"  ✓ Created {image.shape[0]}x{image.shape[1]} test image")
    print(f"  ✓ Created {len(bboxes)} bounding box(es)")
    
    # Step 2: Create Albumentations pipeline
    print("\n[Step 2] Building Albumentations pipeline...")
    pipeline = A.Compose(
        [
            A.Rotate(limit=45, p=1.0, border_mode=0),  # Force rotation
            A.RandomBrightnessContrast(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["bbox_labels"],
        ),
    )
    print("  ✓ Compose([Rotate(limit=45), RandomBrightnessContrast])")
    
    # Step 3: Wrap with PipelineHook
    print("\n[Step 3] Wrapping pipeline with PipelineHook...")
    hooked_pipeline = PipelineHook(pipeline)
    print(f"  ✓ {repr(hooked_pipeline)}")
    
    # Step 4: Run augmentation
    print("\n[Step 4] Running augmentation...")
    pipeline_output = hooked_pipeline(
        image=image,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
    )
    print(f"  ✓ Output image shape: {pipeline_output['image'].shape}")
    print(f"  ✓ Output bboxes: {len(pipeline_output.get('bboxes', []))} box(es)")
    
    # Step 5: Inspect the captured snapshot
    print("\n[Step 5] Inspecting captured snapshot...")
    snapshot = hooked_pipeline.latest_capture
    print(f"  ✓ Transform name: {snapshot.transform_name}")
    print(f"  ✓ Raw bboxes: {len(snapshot.raw_bounding_boxes)}")
    print(f"  ✓ Augmented bboxes: {len(snapshot.augmented_bounding_boxes)}")
    
    if snapshot.augmented_bounding_boxes:
        box = snapshot.augmented_bounding_boxes[0]
        print(f"  ✓ Augmented box coords: ({box.x_min}, {box.y_min}) -> ({box.x_max}, {box.y_max})")
    
    # Step 6: Render the visualization
    print("\n[Step 6] Rendering comparison...")
    renderer = MatplotlibRenderer(figsize=(14, 7))
    output_path = "examples/demo_result.png"
    
    fig = renderer.render(
        snapshot,
        show=False,  # Don't block; just save
        save_path=output_path,
    )
    print(f"  ✓ Saved visualization to: {output_path}")
    
    # Cleanup
    import matplotlib.pyplot as plt
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("Demo complete! Open 'examples/demo_result.png' to see the result.")
    print("=" * 60)


if __name__ == "__main__":
    main()
