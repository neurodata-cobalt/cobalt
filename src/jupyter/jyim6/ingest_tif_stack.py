from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
#import tifffile as tf
import numpy as np
import argparse
import os
import SimpleITK as sitk
import math

def get_channel_resource(rmt, chan_name, coll_name, exp_name, type='image', base_resolution=0, sources=[], datatype='uint16', new_channel=True):
    channel_resource = ChannelResource(chan_name, coll_name, exp_name, type=type,
                                    base_resolution=base_resolution, sources=sources, datatype=datatype)
    if new_channel:
        new_rsc = rmt.create_project(channel_resource)
        return new_rsc

    return channel_resource



def upload_to_boss(rmt, data, channel_resource, resolution=0):
    Z_LOC = 0
    size = data.shape
    for i in range(0, data.shape[Z_LOC], 16):
        last_z = i+16
        if last_z > data.shape[Z_LOC]:
            last_z = data.shape[Z_LOC]
        print(resolution, [0, size[2]], [0, size[1]], [i, last_z])
        rmt.create_cutout(channel_resource, resolution, [0, size[2]], [0, size[1]], [i, last_z], np.asarray(data[i:last_z,:,:], order='C'))

def main():

    parser = argparse.ArgumentParser(description='Ingest a tif stack into The Boss.')
    parser.add_argument('-collection', help='Name of collection to upload tif stack to', type=str)
    parser.add_argument('-experiment', help='Name of experiment to upload tif stack to', type=str)
    parser.add_argument('-channel', help='Name of channel to upload tif stack to. Default is new channel will be created unless otherwise specified. See --new_channel', type=str)
    parser.add_argument('-img_stack', help='Path to image stack to be uploaded', type=str)
    parser.add_argument('-fmt', help='extension of file. can be tif or img', type=str)
    parser.add_argument('--dtype', help='Datatype of ', type=str)
    parser.add_argument('--type', help='Type of upload. Either image or annotation', default='image')
    parser.add_argument('--new_channel', help='True or False depending on if this is an existing channel or new channel needs to be created', default=False)
    parser.add_argument('--source_channel', help='If type of upload is annotation, this argument MUST be provided.', type=str, default=None)
    parser.add_argument('--config', help='Path to configuration file with Boss API token. Default: ~/.intern/intern.cfg', default=os.path.expanduser('~/.intern/intern.cfg'))

    args = parser.parse_args()
    print(args.new_channel)
    rmt = BossRemote(cfg_file_or_dict=args.config)

    type_to_dtype = {'image': 'uint16', 'annotation': 'uint64'}

    if (args.fmt == 'tif'): img = tf.imread(os.path.expanduser(args.img_stack))
    else:
        tmp = sitk.ReadImage(os.path.expanduser(args.img_stack))
        tmp = sitk.RescaleIntensity(tmp, outputMinimum=0, outputMaximum=10000)
        img = sitk.GetArrayFromImage(tmp)
    if args.type == 'annotation' and img.dtype != 'uint64':
        img = np.asarray(img, dtype='uint64')

    coll_name = args.collection
    exp_name = args.experiment
    chan_name = args.channel
    source_chan = []
    if args.source_channel != None:
        source_chan = [args.source_channel]

    # upload image back to boss
    channel_rsc = get_channel_resource(rmt, chan_name, coll_name, exp_name, type=args.type, sources=source_chan, datatype=type_to_dtype[args.type], new_channel=args.new_channel)
    print('uploading {} to boss...'.format(args.type))
    if img.dtype != 'uint64' or img.dtype != 'uint16':
        if args.type == 'image':
            img = img.astype('uint16')
        else:
            img = img.astype('uint64')
    print(img.dtype)
    upload_to_boss(rmt, img, channel_rsc)
    print('upload done! to see upload go to this link: ' + 'http://ben-dev.neurodata.io:8001/ndviz_url/' + coll_name + '/' + exp_name + '/' + chan_name + '/' )

if __name__ == '__main__':
    main()