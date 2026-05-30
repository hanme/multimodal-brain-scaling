from typing import Literal, List
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as mticker


def set_ticks(
    ax: Axes,
    xticks_mode: Literal['linear', 'log', None] = 'log',
    yticks_mode: Literal['linear', 'log', None] = 'log',
    yticks: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    yticklabels: list = None,
    minor_grid: bool = True,
    precision: int = 2,
):
    # Set auto yticks
    if xticks_mode:
        if xticks_mode == 'linear':
            ax.xaxis.set_minor_locator(mticker.LinearLocator(numticks=1))
        elif xticks_mode == 'log':
            ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    
    # Set auto yticks
    if yticks_mode:
        if yticks_mode == 'linear':
            ax.yaxis.set_minor_locator(mticker.LinearLocator(numticks=5))
        elif yticks_mode == 'log':
            ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    
    if minor_grid:
        ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    
    # Set yticks and yticklabels
    if yticks:
        if not yticklabels:
            yticklabels = [f'{t:.{precision}f}' for t in yticks]
        ax.set_yticks(ticks=yticks, labels=yticklabels)
        
    return ax

        
def save_figs(fig, save_dir:str, base_filename:int, dpi:int=300, formats:List[str]=("png", "pdf", "svg")):
    for fmt in formats:
    
        save_fmt_dir = Path(save_dir) / fmt
        if not save_fmt_dir.exists():
            save_fmt_dir.mkdir(parents=True, exist_ok=False)
        
        file_path = save_fmt_dir / f"{base_filename}.{fmt}"
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {file_path}")