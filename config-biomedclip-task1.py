import torchvision.transforms as tt

configs = {
    'model_ver': 'Biomedclip-sam',
    'exp_note': 'breast_same-seq-multi-time',
    'path_csv':'task1_diff_time_same_seq.csv',
    'patch_size':16,
    'embed_dim':768,
    'reg_featureDim': 12,
    'smooth_weight': 2,
    'lr': 3,
    'num_iter': 1000,
    'fm_downsample': 2,
    'feature_size': (14, 14),
    'useSavedPCA': False,
    'foundReg_useMask': False,
    'data_dir':'sample_dataset_dir',
    'save_feature':False,
    'window': True,
    'convex': False,
    'ztrans': False,
    'quantify':False,
    'iter_smooth_num': 5,
    'iter_smooth_kernel': 7,
    'final_upsample': 1,
    'mask': 'slice fill stack',
    'transform': tt.Compose([])
}