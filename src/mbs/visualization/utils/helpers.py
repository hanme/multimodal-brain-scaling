from typing import Callable, List, Tuple, Union

import pandas as pd

def apply_hiearchical_aggregation(df:pd.DataFrame, groupby_cols:List[str], agg_cols:List[str], agg_func:str='mean'):
    """
    Applies hierarchical aggregations to a DataFrame based on specified grouping columns and an aggregation column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame to be aggregated.
        groupby_cols (List[str]): A list of column names to group by for the aggregation.
        agg_cols (List[str]): A list of column names to be aggregated.
        agg_func (str, optional): The aggregation function to apply (default is 'mean').
    """
    if 'subject' not in groupby_cols:
        # If hierarchical aggregation is enabled, we first group by by ROIs and datasets
        groupby_cols_ = list(set(['roi', 'benchmark_name'] + groupby_cols))
        df = df.groupby(groupby_cols_).agg({col: agg_func for col in agg_cols}).reset_index()
    if 'roi' not in groupby_cols:
        # Then we group by datasets
        groupby_cols_ = list(set(['benchmark_name'] + groupby_cols))
        df = df.groupby(groupby_cols_).agg({col: agg_func for col in agg_cols}).reset_index()
        
    df = df.groupby(groupby_cols).agg({col: agg_func for col in agg_cols}).reset_index()
    
        
    return df
