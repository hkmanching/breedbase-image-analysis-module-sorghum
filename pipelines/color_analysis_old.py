import numpy as np
import cv2

def get_color_stats_by_label(labeled_mask, image, image_filename):
    rows = []
    labels = np.unique(labeled_mask)
    labels = labels[labels != 0]  # Exclude background (label 0)

    for label_id in labels:
        stats_dict = {}
        # mask_indices = (labeled_mask == (i + 1)).astype(np.uint8) * 255
        mask_indices = np.where(labeled_mask == label_id)
        if len(mask_indices[0]) == 0:
            continue

        # Extract pixel values
        # pixels_bgr = image_bgr[mask_indices]
        # pixels_rgb = pixels_bgr[:, ::-1]  # Convert to RGB for LAB
        # pixels_lab = rgb2lab(pixels_rgb / 255.0)
        pixels_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[mask_indices]
        pixels_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[mask_indices]
        pixels_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[mask_indices]

        color_spaces = {
            'RGB': pixels_rgb,
            'LAB': pixels_lab,
            'HSV': pixels_hsv
        }

        for space_name, pixels in color_spaces.items():
            for i, ch_name in enumerate(['C1', 'C2', 'C3']):
                channel_data = pixels[:, i]
                mode_result = stats.mode(channel_data)
                mode_val = mode_result.mode
                stats_result = {
                            'mean': float(np.mean(channel_data)),
                            'max': int(np.max(channel_data)),
                            'min': int(np.min(channel_data)),
                            'mode': int(mode_val),
                            'median': float(np.median(channel_data)),
                            'std': float(np.std(channel_data))
                        }

                channel_name = space_name[i]
                row = {
                    'image_filename': image_filename,
                    'label': f'Seed_{label_id}',
                    'color_space': space_name,
                    'channel': f'Channel_{channel_name}',
                    **stats_result
                }
                rows.append(row)

    return rows