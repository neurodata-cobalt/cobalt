# register_brain.py
import sys
import os
import ndreg
from ndreg import preprocessor, registerer
import SimpleITK as sitk
import numpy as np
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import skimage
import argparse
import time

def downsample(img, res=3):
    out_spacing = np.array(img.GetSpacing()) * (2.0**res)
    img_ds = skimage.measure.block_reduce(sitk.GetArrayViewFromImage(img),
                                          block_size=(2,2,2), func=np.mean)
    for i in range(res - 1):
        img_ds = skimage.measure.block_reduce(img_ds, block_size=(2,2,2), func=np.mean)
    img_ds_sitk = sitk.GetImageFromArray(img_ds)
    img_ds_sitk.SetSpacing(out_spacing)
    return img_ds_sitk

def main():
    t_start_overall = time.time()
    parser = argparse.ArgumentParser(description='Register a brain in the BOSS and upload it back in a new experiment.')
    parser.add_argument('--collection', help='Name of collection to upload tif stack to', type=str)
    parser.add_argument('--experiment', help='Name of experiment to upload tif stack to', type=str)
    parser.add_argument('--channel', help='Name of channel to upload tif stack to. Default is new channel will be created unless otherwise specified. See --new_channel', type=str)
    parser.add_argument('--image_orientation', help='Orientation of brain image. 3-letter orientation of brain. For example can be PIR: Posterior, Inferior, Right.', type=str)
    parser.add_argument('--outdir', help='set the directory in which you want to save the intermediates. default is ./{collection}_{experiment}_{channel}_reg', type=str, default=None)
    parser.add_argument('--resume', help='If steps of the registration completed, pick up from where you left off', action='store_true')
    parser.add_argument('--config', help='Path to configuration file with Boss API token. Default: ~/.intern/intern.cfg', default=os.path.expanduser('~/.intern/intern.cfg'))

    args = parser.parse_args()
    # conversion factor from mm to um
    mm_to_um = 1000.0

    # download image
    rmt = BossRemote(cfg_file_or_dict=args.config)
    # resolution level from 0-6
    resolution_image = 3
    # resolution in microns
    resolution_atlas = 50
    # atlas orientation known
    orientation_atlas = 'pir'
    # ensure outdir is default value if None
    if args.outdir is None:
        outdir = '{}_{}_{}_reg/'.format(args.collection, args.experiment, args.channel)
        ndreg.dirMake(outdir)
    else: outdir = args.outdir

    # downloading image
    if not args.resume or not os.path.isfile(outdir + '{}_{}_{}um.img'.format(args.experiment, args.channel, resolution_atlas)): 
        print('downloading experiment: {}, channel: {}...'.format(args.experiment, args.channel))
        t1 = time.time()
        img = ndreg.download_image(rmt, args.collection, args.experiment, args.channel)
        print("time to download image at res {} um: {} seconds".format(img.GetSpacing()[0] * mm_to_um, time.time()-t1))

        # download atlas
        print('downloading atlas...')
        t1 = time.time()
        atlas = ndreg.download_ara(rmt, resolution_atlas, type='average')
        print("time to download atlas at {} um: {} seconds".format(resolution_atlas, time.time()-t1))

        # resample image to match atlas spacing
        print('downsampling image...')
        t1 = time.time()
#        downsample_factor = 3
#        img_ds = downsample(img, res=downsample_factor)
        img = ndreg.imgResample(img, [resolution_atlas/mm_to_um]*3)
        print("time to downsample image: {} seconds".format(time.time() - t1))
        # save 
        sitk.WriteImage(img, outdir + '{}_{}_{}um.img'.format(args.experiment, args.channel, resolution_atlas))
    else:
        img = ndreg.imgRead(outdir + '{}_{}_{}um.img'.format(args.experiment, args.channel, resolution_atlas))
        atlas = ndreg.download_ara(rmt, resolution_atlas, type='average')
        

    if not args.resume or not os.path.isfile(outdir + '{}_{}_bias_corrected.img'.format(args.experiment, args.channel)):
        # do the bias correction
        print("creating mask and correcting bias field in target...")
        t1 = time.time()
        mask_dilation_radius = 10 # voxels
        mask = sitk.BinaryDilate(preprocessor.create_mask(img, use_triangle=True), mask_dilation_radius)
        img_bc, bias = preprocessor.correct_bias_field(img, scale=0.25, spline_order=4, mask=mask,
                                                 num_control_pts=[5,5,5],
                                                 niters=[500, 500, 500, 500])
        print("time to bias correct image: {} seconds".format(time.time() - t1))
        # save bias corrected image
        sitk.WriteImage(img_bc, outdir + '{}_{}_bias_corrected.img'.format(args.experiment, args.channel))
    else:
        img_bc = ndreg.imgRead(outdir + '{}_{}_bias_corrected.img'.format(args.experiment, args.channel))
    
    # reorient the image to match atlas
    print("image size: {}, image orientation: {}".format(img_bc.GetSize(), args.image_orientation))
    print("atlas size: {}, atlas orientation: {}".format(atlas.GetSize(), orientation_atlas))

    img_bc = ndreg.imgReorient(img_bc, args.image_orientation, orientation_atlas)

    print("normalizing atlas and target...")
    t1 = time.time()
    atlas_n = sitk.Normalize(atlas)
    img_bc_n = sitk.Normalize(img_bc)
    print("time taken to normalize atlas and target: {} seconds".format(time.time()-t1))


    # affine registration
    print("performing affine registration...")
    t1 = time.time()
    final_transform = registerer.register_affine(atlas_n, 
                                                img_bc_n,
                                                learning_rate=1e-1,
                                                grad_tol=4e-6,
                                                use_mi=False,
                                                iters=50,
                                                shrink_factors=[4,2,1],
                                                sigmas=[0.4, 0.2, 0.1],
                                                verbose=False)

    atlas_affine = registerer.resample(atlas, final_transform, img_bc, default_value=ndreg.imgPercentile(atlas,0.01))
    target_affine = registerer.resample(img_bc, final_transform.GetInverse(), atlas, default_value=ndreg.imgPercentile(img_bc,0.01))
    print("time taken for affine registration: {} seconds".format(time.time() - t1))
    # save affine registered image
    sitk.WriteImage(atlas_affine, outdir + 'atlas_to_{}_affine.img'.format(args.experiment))
    sitk.WriteImage(target_affine, outdir + '{}_to_atlas_affine.img'.format(args.experiment))

    # whiten the images
    print("whitening images...")
    t1 = time.time()
    atlas_affine_w = sitk.AdaptiveHistogramEqualization(atlas_affine, [10,10,10], alpha=0.25, beta=0.25)
    img_bc_w = sitk.AdaptiveHistogramEqualization(img_bc, [10,10,10], alpha=0.25, beta=0.25)
    print('time to whiten atlas and target: {}'.format(time.time() - t1))

    # lddmm code
    print("beginning LDDMM parameter sweep")
    t1 = time.time()
    e = 5e-3
    s = 0.1
    atlas_lddmm, field, inv_field = registerer.register_lddmm(affine_img=sitk.Normalize(atlas_affine_w), 
                                                              target_img=sitk.Normalize(img_bc_w),
                                                              alpha_list=[0.05], 
                                                              scale_list = [0.0625, 0.125, 0.25, 0.5, 1.0],
                                                              epsilon_list=e, sigma=s,
                                                              min_epsilon_list=e*1e-6,
                                                              use_mi=False, iterations=50, verbose=True,
                                                              out_dir=outdir + 'lddmm')
    print("time taken for LDDMM: {} seconds".format(time.time()-t1))
    end_time = time.time()
    print("Overall time taken through all steps: {} seconds ({} minutes)".format(end_time - t_start_overall, (end_time - t_start_overall)/60.0))

if __name__ == "__main__":
    main()
