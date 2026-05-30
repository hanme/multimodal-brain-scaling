from typing import List, Tuple, Union

import numpy as np
import seaborn as sns

def rgb2hex(rgb: Union[List[int], Tuple[int, int, int]]) -> str:
    """Convert RGB to hex color."""
    rgb = np.array(rgb)
    if np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError('RGB values should be in the range [0, 255]')
    if len(rgb) != 3:
        raise ValueError('RGB should have 3 values')
    
    if np.any( (0 < rgb) & (rgb < 1) ):
        rgb = (rgb * 256).astype(int)
     
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'.upper()

def hex2rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB."""
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def mix_hex_colors(color1_hex:str, color2_hex:str):
    color1_rgb = hex2rgb(color1_hex)
    color2_rgb = hex2rgb(color2_hex)
    color3_rgb = tuple((np.array(color1_rgb) + np.array(color2_rgb)) // 2)
    return rgb2hex(color3_rgb)

## Model colors
palette_models = ["#FDC5F5","#F7AEF8", "#B388EB", "#8093F1", "#79b8f4", "#72DDF7"]
palette_models = sns.color_palette(palette_models, n_colors=6, as_cmap=False)

## Dataset colors
palette_datasets = ["#4F000B", "#720026", "#CE4257", '#E66054', "#FF7F51", "#FF9B54"]
palette_datasets = sns.color_palette(palette_datasets, n_colors=6, as_cmap=False)

## Region colors
color_start = '#40E0D0' # turquoise cyan
color_end = '#008b8b' # dark cyan
palatte_regions  = sns.color_palette(f"blend:{color_start},{color_end}", n_colors=5, as_cmap=False)


## Sample size colors
color = '#008b8b' # dark cyan
palette_samples  = sns.light_palette(color, as_cmap=True)


COLORS = {
    'green_1': '#418243',
    'green_2': '#54a955',
    'green_3': '#69b87b',
    'blue_1': '#4b9dc6',
    'blue_2': '#4c9ccb',
    'blue_3': '#87c2de',
    'red_1': '#ce3631',
    'cyan_dark': '#008b8b',
    'cyan_light': '#E0FFFF',
    'cyan_turquoise': '#40E0D0',
    'cyan_peacock': '#004958',
}

COLOR_PALETTES = {
    'models': palette_models,
    'datasets': palette_datasets,
    'regions': palatte_regions,
    'samples': palette_samples,
}


BENCHMARK_COLORS = {
    # --------------------
    # fMRI — blue end
    # --------------------
    "things_fmri": "#1F4E79",   # deep blue
    "nsd_func1pt8mm_individualROIs": "#2E6F95",
    "nsd": "#2E6F95",           # alias

    # --------------------
    # EEG / MEG — middle (blue-green / teal)
    # --------------------
    "things_eeg1": "#2A9D8F",
    "things_eeg2": "#4FB3A2",
    "things_meg": "#3CA6A6",

    # --------------------
    # Electrophysiology (EP) — green end
    # --------------------
    "bs_fz": "#2E7D32",
    "bs_mh": "#4CAF50",
    "tvsd": "#66BB6A",

    # --------------------
    # Aggregate
    # --------------------
    # "benchmark_average": "#9E9E9E",
    "benchmark_average": "#1b263b",
}

# ARCHITECUTURE_FAMILY_COLORS = {
#     'AlexNet': "#2d0b59",
#     'ConvNeXt': "#f77f00",
#     'CORnet-S': "#6a0572",
#     'ViT': "#b12a90",
#     'EfficientNet': "#e63946",
#     'ResNet': "#7a3e00",
#  }


ARCHITECUTURE_FAMILY_COLORS = {
    'AlexNet': "#b12a90",
    'ConvNeXt': "#f77f00",
    'CORnet-S': "#2d0b59",
    'ViT': "#e63946",
    'EfficientNet': "#7a3e00",
    'ResNet': "#6a0572",
 }

# READOUT_COLORS = {
#     'Linear (Single-Subject)': '#ef476f',
#     'Attention (Single-Subject)': '#ffd166',
#     'Attention (Multi-Subject)': '#06d6a0',
# }
READOUT_COLORS = {
    'linear': '#ef476f',
    'attention_individual': '#ffd166',
    'attention_shared': '#06d6a0',
}