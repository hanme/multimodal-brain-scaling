from typing import Dict, List

BENCHMARK_NAME_MAPPING: Dict[str, str] = {
    'bs_fz': 'FZ-EP',
    'bs_mh': 'MH-EP',
    'tvsd': 'TVSD-EP',
    'things_fmri': 'T-fMRI',
    'nsd_func1pt8mm_individualROIs': 'NSD-fMRI',
    'nsd': 'NSD-fMRI',
    'things_eeg1': 'T-EEG1',
    'things_eeg2': 'T-EEG2',
    'things_meg': 'T-MEG',
    'benchmark_average': 'Average'
}

METRICS_ALL: List[str] = [
    "pearsonr_nc",
    "pearsonr",
    "cv_score",
    "rsa_c_train",
    "rsa_c_test",
    "cka_c_train",
    "cka_c_test",
    
    "rsa_ve_train",
    "rsa_ve_test",
    "cka_ve_train",
    "cka_ve_test",
]

METRICS_COMPACT: List[str] = [
    "pearsonr_nc",
    "rsa_c_test",
    "cka_c_test",
    "rsa_ve_test",
    "cka_ve_test",
]

GENERIC_GROUPING_COLUMNS: List[str] = [
    'model_id', 
    'backbone_source', 
    'backbone_arch',
    'backbone_n_params', 
    'backbone_macs',
    'backbone_flops', 
    'backbone_arch_family',
    'pretraining_dataset', 
    'pretraining_n_samples',
    'pretraining_total_flops_estimate',
]