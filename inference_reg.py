import glob
import sys, os
import numpy as np
from sympy import use
import torch
from scipy.ndimage import zoom
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config
import torchvision.transforms as tt
import nibabel as nib
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from skimage.measure import label, regionprops
from skimage import morphology
import json
import scipy.ndimage
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom
from sam import SamPredictor, sam_model_registry
from transformers import AutoModel, CLIPModel

from convex_adam_utils import *
import sys
import einops
from utils.img_operations import remove_uniform_intensity_slices, reconstruct_image, CT_normalize, clip_and_normalize_image, pca_lowrank_transform, MR_normalize, jacobian_determinant, compute_95_hausdorff_distance, compute_label_wise_95hd
from utils.img_operations import extract_lung_mask
from utils.convexAdam_3D import convex_adam_3d, convex_adam_3d_w0, convex_adam_3d_interSmooth, convex_adam_3d_param, convex_adam_3d_param_dataSmooth
from utils.data_utils import get_files_mrct
from scipy.ndimage import laplace, gaussian_filter
import utils.img_operations as img_op
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from os import path
from PIL import Image
import matplotlib.pyplot as plt
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
if XFORMERS_ENABLED:
    from xformers.ops import memory_efficient_attention, unbind
import csv
import pandas as pd
import nrrd
import cfg
args = cfg.parse_args()
import importlib.util
import argparse
from utils import img_operations
"""
FILE NOTE:

"""

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module.configs

class foundReg:
    def __init__(self, configs, device_id=0):
        self.device_id = device_id
        self.model_ver = configs['model_ver']
        self.patch_size = configs['patch_size']
        self.transform = configs['transform']
        self.embed_dim = configs['embed_dim']
        self.reg_featureDim = configs['reg_featureDim']    
        self.patch_margin = 10
        self.src_slice_num = 2
        self.patch_grid_h, self.patch_grid_w = 8, 8
        self.slice_step = 5
        self.model = self.load_model()
        feat_size = configs['feature_size']
        self.img_size = (self.patch_size*feat_size[0], self.patch_size*feat_size[1])  #set size
        self.num_iter = configs['iter_smooth_num'] 
        self.lr = configs['lr']
        self.smooth_weight = configs['smooth_weight']
        self.save_feature = configs['save_feature']
        print('learning rate', self.lr)

        self.feature_height = self.img_size[0] // self.patch_size
        self.feature_width = self.img_size[1] // self.patch_size


    def extract_encoder_feature(self, input_array):
        assert len(input_array.shape) == 3  # 2D image
        """flipping the input if needed"""
        # for dino-v2, the input size should be divided by 14
        input_rgb_array = input_array[np.newaxis, :, :, :]
        input_tensor = torch.Tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]))
        input_tensor = self.transform(input_tensor)
        print('input_tensor shape:',input_tensor.shape, input_tensor.max(),input_tensor.min())
        if self.model_ver in ('dino-v2', 'dino-v3'):
            feature_array = self.model.forward_features(input_tensor.to(device=torch.device('cuda', self.device_id)))[
                'x_norm_patchtokens'].detach().cpu().numpy()
        elif self.model_ver =='mricore':
            feature_array= self.model.image_encoder(input_tensor.to(device=torch.device('cuda', self.device_id)), before_neck=True).detach().cpu().numpy()
            feature_array = einops.rearrange(feature_array, '1 h w  c-> 1 (h w) c')
        elif self.model_ver in ['sam','medsam','sslsam']:
            feature_array= self.model.image_encoder(input_tensor.to(device=torch.device('cuda', self.device_id))).detach().cpu().numpy()*10
            feature_array = einops.rearrange(feature_array, '1 c h w -> 1 (h w) c')
        elif self.model_ver == 'Biomedclip-sam':
            feature_array = self.model(pixel_values=input_tensor.to(device=torch.device('cuda', self.device_id))).last_hidden_state.detach().cpu().numpy()
            feature_array = feature_array[:,1:,:]                                  
        print('feature_tensor shape:',feature_array.shape, feature_array.max(),feature_array.min())
        del input_tensor
        return feature_array


    def case_inference(self, mov_arr, fix_arr, orig_img_shape, aff_mov,
                       mask_fixed=None, mask_moving=None, fix_case_id='noID', mov_case_id = 'noID', disp_init=None, grid_sp_adam=1,foundReg_useMask=True):

        assert len(mov_arr.shape) == 3

        """prepcocessing and feature extraction"""
        mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape, mask_fixed_arr, mask_moving_arr = self.case_preprocess(mov_arr, fix_arr)


        print('preprocessed moving and fixed image, shape', mov_arr.shape, fix_arr.shape)
        gap = 6
        if  self.model_ver =='MIND':
            mov_feature =  MINDSSC(torch.from_numpy(mov_arr).unsqueeze(0).unsqueeze(0).float().to(device=torch.device('cuda', self.device_id)),3,2).detach().cpu().numpy()*10
            fix_feature = MINDSSC(torch.from_numpy(fix_arr).unsqueeze(0).unsqueeze(0).float().to(device=torch.device('cuda', self.device_id)),3,2).detach().cpu().numpy()*10
            feat_sliceNum = mov_arr.shape[2]
            
            print('encoded moving image, feature shape:', mov_feature.shape)
            print('encoded fixed image, feature shape:', fix_feature.shape)
            print(mov_feature.max(),mov_feature.min(),fix_feature.max(),fix_feature.min())
            mov_pca_rescaled = np.transpose(mov_feature.squeeze(0), (1, 2, 3, 0))
            fix_pca_rescaled = np.transpose(fix_feature.squeeze(0), (1, 2, 3, 0))
        else:
            mov_feature = self.encode_3D_gap(mov_arr, gap=gap)
            fix_feature = self.encode_3D_gap(fix_arr, gap=gap)
            feat_sliceNum = self.slice_num
            print('encoded moving image, feature shape:',mov_feature.shape)
            print('encoded fixed image, feature shape:',fix_feature.shape)

            if self.save_feature:
                os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
                np.save(os.path.join(output_dir, 'features', mov_case_id + '_feat_org.npy'), mov_feature)
                np.save(os.path.join(output_dir, 'features', fix_case_id + '_feat_org.npy'), fix_feature)


            """PCA reduce dimension"""
            #only features inside the mask
            if foundReg_useMask:
                # reshape to model output
    
                mask_fixed_arr = resize(mask_fixed_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                    anti_aliasing=True)
                mask_moving_arr = resize(mask_moving_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                     anti_aliasing=True)
                mask_fixed_arr = np.where(mask_fixed_arr > 0.99, 1.0, 0)
                mask_moving_arr = np.where(mask_moving_arr > 0.99, 1.0, 0)
                # fixImg_1dim_threshold = nib.Nifti1Image(mask_fixed_arr, aff_mov)
                # nib.save(fixImg_1dim_threshold, os.path.join(output_dir, 'vis',case_list[i] + '_threshold.nii.gz'))
    
                # print('mask shape', mask_moving_arr.shape, mask_fixed_arr.shape)
                # print('feature  shape', mov_feature.shape, fix_feature.shape)
                mask_moving_arr = mask_moving_arr.flatten().astype(bool)
                mask_fixed_arr = mask_fixed_arr.flatten().astype(bool)
                mov_feature = mov_feature[mask_moving_arr, :]
                fix_feature = fix_feature[mask_fixed_arr, :]
    
    
            print('Starting PCA to reduce dimension')
            all_features = np.concatenate([mov_feature,fix_feature], axis=0)
            print('all features shape', all_features.shape, 'mask sum', mask_moving_arr.sum(), mask_fixed_arr.sum())
            pca_start_time = time.time()
            # object_pca = PCA(n_components=self.reg_featureDim) #what is SVD solver?
            # reduced_patches = object_pca.fit_transform(all_features)
            if configs['useSavedPCA']:
                reduced_patches = np.dot(all_features, PCA_matrix)
                eigenvalues = np.zeros(24)
            else:
                reduced_patches, eigenvalues = pca_lowrank_transform(all_features, self.reg_featureDim)

            print('PCA finished in {}, splitting features'.format(time.time()-pca_start_time))
            print('PCA finsihed feature shape:',reduced_patches.shape)
    
            if foundReg_useMask:
                mov_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
                fix_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
                mov_pca[mask_moving_arr, :] = reduced_patches[:mask_moving_arr.sum(), :]
                fix_pca[mask_fixed_arr, :] = reduced_patches[mask_moving_arr.sum():, :]
                mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
                fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
            else:
                mov_pca = reduced_patches[:feat_sliceNum * self.feature_height * self.feature_width, :]
                fix_pca = reduced_patches[feat_sliceNum * self.feature_height * self.feature_width:, :]
                mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
                fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
    
            eigenvalue_array.append(eigenvalues[:24])
    
            print('reshaping to original image shape')
            mov_pca_rescaled = resize(mov_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                       anti_aliasing=True)
            fix_pca_rescaled = resize(fix_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                       anti_aliasing=True)
    
    
        #plug in the slices to keep, the rest are 0
        mov_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                          dtype='float32')
        fix_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                      dtype='float32')

        mov_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = mov_pca_rescaled
        fix_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = fix_pca_rescaled


        

        if self.save_feature:
            """save copy of 1 channel feature for vis"""
            for channel in range(1):
                mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,channel*3:channel*3+3]
                fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,channel*3:channel*3+3]
                movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
                fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
                os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
                nib.save(movImg_1dim, os.path.join(output_dir, 'vis', mov_case_id + '_mov_{}.nii.gz'.format(channel)))
                nib.save(fixImg_1dim, os.path.join(output_dir, 'vis', fix_case_id + '_fix_{}.nii.gz'.format(channel)))
            
            os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
            np.save(os.path.join(output_dir, 'features', mov_case_id + '_feat_pca.npy'), mov_fullImg_pca_rescaled)
            np.save(os.path.join(output_dir, 'features', fix_case_id + '_feat_pca.npy'), fix_fullImg_pca_rescaled)

        """ConvexAdam optimization"""
        print('starting ConvexAdam optimization')
        print('moving image pca size',mov_fullImg_pca_rescaled.shape)
        disp = convex_adam_3d_param(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "SSD", grid_sp_adam=grid_sp_adam,
                                               lambda_weight=configs['smooth_weight'], selected_niter=configs['num_iter'], lr=configs['lr'], disp_init=disp_init,
                                                iter_smooth_kernel = configs['iter_smooth_kernel'],
                                                iter_smooth_num = configs['iter_smooth_num'], end_smooth_kernel=1,final_upsample=configs['final_upsample'])

        return disp

    def case_preprocess(self, mov_arr, fix_arr):
        assert len(mov_arr.shape) == 3
        assert len(fix_arr.shape) == 3

        pad_indices = []
        filtered_image_data, slices_to_keep_indices = remove_uniform_intensity_slices(fix_arr)
        pad_indices.append(slices_to_keep_indices)
        fix_arr = filtered_image_data
        mov_arr = mov_arr[:, :, slices_to_keep_indices]

        orig_chunked_shape = fix_arr.shape

        mask_fixed = np.where(fix_arr > 0.05, 1.0, 0)
        mask_moving = np.where(mov_arr > 0.05, 1.0, 0)


        filtered_z = fix_arr.shape[2]


        mask_fixed = np.zeros_like(fix_arr)
        mask_moving = np.zeros_like(mov_arr)
        for slice_idx in range(fix_arr.shape[2]):
            mask_fixed[:, :, slice_idx] = extract_lung_mask(fix_arr[:, :, slice_idx], threshold_value=0.05)
            mask_moving[:, :, slice_idx] = extract_lung_mask(mov_arr[:, :, slice_idx], threshold_value=0.005)


        return mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape , mask_fixed, mask_moving

    def load_model(self,model_name='dino-v2'):
        """load model"""

        if self.model_ver =='MIND':
            return None
        elif self.model_ver == 'dino-v3':
            self.patch_size = 16
            self.embed_dim = 768
            model_name =  'dinov3_vitb16'
            DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
            DINOV3_LOCATION = './dinov3'
            model = torch.hub.load(
                repo_or_dir=DINOV3_LOCATION,
                model=model_name,
                source="local",
                weights = './dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            )
        elif self.model_ver=='dino-v2':
            conf_fn = '{0:s}/dinov2/configs/eval/vitl14_reg4_pretrain'.format(sys.path[0])
            model_fn = 'models/dinov2/dinov2_vitl14_reg4_pretrain.pth'
            model_url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'
            self.patch_size = 14
            self.embed_dim = 1024
            # Check if model file exists, download if not
            if not os.path.exists(model_fn):
                import urllib.request
                os.makedirs(os.path.dirname(model_fn), exist_ok=True)
                print(f"Downloading model from {model_url} to {model_fn}...")
                urllib.request.urlretrieve(model_url, model_fn)
                print("Download complete.")
            else:
                print("DINOv2 model found.")
            conf = load_and_merge_config(conf_fn)
            model = build_model_for_eval(conf, model_fn)
        elif self.model_ver =='sam':
            model = sam_model_registry["vit_b"](args,checkpoint=os.path.join("sam_vit_b_01ec64.pth")).eval()
        elif self.model_ver =='medsam':
            model = sam_model_registry["vit_b"](args,checkpoint=os.path.join('medsam_vit_b.pth')).eval()
        elif self.model_ver =='sslsam':
            model = sam_model_registry["vit_b"](args,checkpoint=os.path.join("sam_vit_b_01ec64.pth"),num_classes=2)
            pretrained_encoder_checkpoint = torch.load(os.path.join('sslsam.pth'),map_location='cpu')
            new_state_dict = {k.replace("module.", ""): v for k, v in pretrained_encoder_checkpoint.items()}
            model.image_encoder.load_state_dict(new_state_dict,strict = False)
        elif self.model_ver =='mricore':
            print(args)
            model = sam_model_registry["vit_b"](args,checkpoint=os.path.join('teacher_checkpoint.pth')).eval()
        elif self.model_ver =='Biomedclip-sam':
            model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
            state_dict = torch.load("medclip-samv2-model/pytorch_model.bin", map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model = model.vision_model
            
        model.to(device=torch.device('cuda', self.device_id)).eval()
        return model

    def encode_3D_gap(self, input_arr, gap=3):


        imageH, imageW, slice_num = input_arr.shape

        """new uniform resize"""
        feature_height = self.feature_height
        feature_width = self.feature_width
        self.slice_num = slice_num


        input_arr = resize(input_arr, (feature_height*self.patch_size, feature_width*self.patch_size, slice_num), anti_aliasing=True)

        print(self.patch_size)
        print(feature_height, feature_width, slice_num)
        print('resized input shape', input_arr.shape)

        # 3D image into 2D model, stack each slices feature
        img_feature = np.zeros([feature_height * feature_width, slice_num, self.embed_dim])
        encoding_slice_idx = np.arange(0, slice_num-1, gap).tolist()
        encoding_slice_idx.append(slice_num-1)

        prev_slice = 0
        for slice_id in encoding_slice_idx:
            input_slice = input_arr[:, :, slice_id, np.newaxis]
            input_slice = np.repeat(input_slice, 3, axis=2)
            featrure = self.extract_encoder_feature(input_slice)
            featrure = einops.rearrange(featrure, '1 n c -> n c')
            print("\rslice id:{} feature shape:{} ".format(slice_id, featrure.shape), end="")
            img_feature[:, slice_id, :] = featrure

            #interpolating the feature of the skipped slices
            if slice_id > 0 and slice_id < slice_num-1:
                for i in range(1, gap):
                    slice_id_gap = slice_id - i
                    if slice_id_gap >= 0:
                        featrure_gap = (featrure * (gap - i) + img_feature[:, prev_slice, :] * i) / gap
                        img_feature[:, slice_id_gap, :] = featrure_gap
            elif slice_id == slice_num-1:
                last_gap = slice_num - encoding_slice_idx[-2]
                for i in range(1, last_gap):
                    slice_id_gap = slice_num - i
                    featrure_gap = (featrure * (last_gap - i) + img_feature[:, prev_slice, :] * i) / last_gap
                    img_feature[:, slice_id_gap, :] = featrure_gap
            prev_slice = slice_id

        img_feature = img_feature.reshape([feature_height * feature_width * slice_num, self.embed_dim])

        return img_feature


    def extract_slice_feature(self, input_arr_orig, mask=True):

        """input single slice 2d, output the feature of that slice"""

        input_arr = resize(input_arr_orig, (self.feature_height*self.patch_size, self.feature_width*self.patch_size), anti_aliasing=True)
        if mask:
            input_arr_masksize = resize(input_arr_orig, (self.feature_height, self.feature_width), anti_aliasing=True)
            pca_mask = extract_lung_mask(input_arr_masksize).flatten().astype(bool)

        input_slice = input_arr[:, :, np.newaxis]
        input_slice = np.repeat(input_slice, 3, axis=2)
        featrure = self.extract_encoder_feature(input_slice)

        featrure = einops.rearrange(featrure, '1 n c -> n c')

        if mask:
            return featrure, pca_mask
        return featrure, np.ones(featrure.shape[0], dtype=bool)


if __name__ == '__main__':

    time_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Path to Python config file')
    args2 = parser.parse_args()
    configs = load_config(args2.cfg)
    save_feature = configs['save_feature']
    dataset_dir = configs['data_dir']
    output_dir_0 = f'output/foundReg-model-{configs["model_ver"]}-{configs["smooth_weight"]}smooth-{configs["num_iter"]}iter-itersmoothK{configs["iter_smooth_kernel"]}R{configs["iter_smooth_num"]}-lr3-fmd1-fmsize{configs["feature_size"][0]}-noconvex' #
    print('output_dir_0', output_dir_0)


    if configs['useSavedPCA']:
        PCA_matrix = np.load('/sample_dir/pca_matrix_AMOS_150x129_mask.npy')


    os.makedirs(output_dir_0, exist_ok=True)

    #save config as json
   

    # Remove 'transform' temporarily for logging or saving
    configs_to_save = {k: v for k, v in configs.items() if k != 'transform'}
    with open(os.path.join(output_dir_0,'config_log.json'), 'w') as f:
        json.dump(configs_to_save, f, indent=2)
    
    # Initialize an empty list to hold the rows
    pair_list = []
    # Read the CSV file
    with open(path.join(dataset_dir, configs['path_csv']), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            pair_list.append(row)

    pair_list = [pair_list[i] for i in [1,2,3]] #pair_list[[1:]]
    #pair_list = pair_list[31:]

    quantify = configs['quantify']

    foundReg = foundReg(configs)

    exp_note = configs['exp_note']

    print('exp_note', exp_note)

    output_dir = os.path.join(output_dir_0, exp_note)
    os.makedirs(output_dir, exist_ok=True)

    eigenvalue_array = []

    print('exp_note', exp_note)

    for i, pair in enumerate(pair_list):
        print('case', i)

        moving_fn = pair[0] #template is to be applied to other cases
        fixed_fn = pair[1]

        fixed_basename = os.path.basename(fixed_fn)
        fixed_basename = fixed_basename.split('.')[0]
        moving_basename = os.path.basename(moving_fn)
        moving_basename = moving_basename.split('.')[0]

        img_fixed = nib.load(path.join(dataset_dir, 'img_processed', fixed_fn))
        img_moving = nib.load(path.join(dataset_dir, 'img_processed', moving_fn))
        

        arr_fixed = img_fixed.get_fdata()
        arr_moving = img_moving.get_fdata()
        print('image shape',arr_fixed.shape)

        aff_mov = img_moving.affine

        # Load NRRD segmentations
        
        seg_fixed_path = path.join(dataset_dir, 'seg', fixed_fn.replace('.nii.gz', '.seg.nrrd'))
        seg_moving_path = path.join(dataset_dir, 'seg', moving_fn.replace('.nii.gz', '.seg.nrrd'))
        if os.path.exists(seg_fixed_path) and os.path.exists(seg_moving_path):
            seg_fixed, _ = nrrd.read(seg_fixed_path)
            seg_moving, _ = nrrd.read(seg_moving_path)
            print('mask shape',seg_fixed.shape)
        else:
            seg_fixed = np.zeros_like(arr_fixed)
            seg_moving = np.zeros_like(arr_moving)
            


        H,W,D = arr_moving.shape
        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3)
        disp_init = identity.numpy()

        disp_init = None
        load_note = None
        # load_note = 'cls_select'
        if load_note is not None:
            disp_init = nib.load(os.path.join(output_dir_0, load_note, i + '_disp_{}.nii.gz'.format(load_note))).get_fdata()
            disp_init = np.moveaxis(disp_init, 3, 0)[np.newaxis, :, :, :, :]

        disp = foundReg.case_inference(arr_moving, arr_fixed, arr_moving.shape, aff_mov, fix_case_id=fixed_basename,mov_case_id=moving_basename, 
                                      disp_init=disp_init, grid_sp_adam=configs['fm_downsample'], foundReg_useMask=configs['foundReg_useMask'])
        
        #save disp
        disp_img = nib.Nifti1Image(disp, aff_mov)
        # nib.save(disp_img, os.path.join(output_dir, '{}_disp_{}.nii.gz'.format(fixed_basename, exp_note)))
        nib.save(disp_img, os.path.join(output_dir, '{}_to_{}_disp_{}.nii.gz'.format(
            moving_basename, fixed_basename, exp_note)))

        disp = np.moveaxis(disp, 3, 0)

        #warp moving image
        D, H, W = arr_moving.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        warped_image = map_coordinates(arr_moving, identity + disp, order=0)
        warped_image = nib.Nifti1Image(warped_image, aff_mov)
        # nib.save(warped_image, os.path.join(output_dir, '{}_warped_{}.nii.gz'.format(fixed_basename, exp_note)))
        nib.save(warped_image, os.path.join(output_dir, '{}_to_{}_warped_{}.nii.gz'.format(
            moving_basename, fixed_basename, exp_note)))

    print('time elapsed', time.time() - time_start, 'exp_note', exp_note)
    print(output_dir_0)

