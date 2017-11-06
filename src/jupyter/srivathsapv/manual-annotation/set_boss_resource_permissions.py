import argparse
import configparser
import sys

from bossmeta import *


def get_boss_config(boss_config_file):
    config = configparser.ConfigParser()
    config.read(boss_config_file)
    token = config['Default']['token']
    boss_url = ''.join(
        (config['Default']['protocol'], '://', config['Default']['host']))
    return token, boss_url


def main():
    parser = argparse.ArgumentParser(
        description='Set permissions on collection/experiment/channel')
    parser.add_argument('--group', type=str,
                        help='Boss group made up of boss users')
    parser.add_argument('--collection', type=str, help='Boss collection name')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Boss experiment name')
    parser.add_argument('--channel', type=str, default=None,
                        help='NDStore / OCP channel')
    parser.add_argument('--boss_config_file', type=str,
                        default='neurodata.cfg', help='Boss config file')

    perm_type_parser = parser.add_mutually_exclusive_group(required=False)
    perm_type_parser.add_argument(
        '--read_only', dest='perm_type_admin', action='store_false')
    perm_type_parser.add_argument(
        '--admin', dest='perm_type_admin', action='store_true')
    parser.set_defaults(perm_type_admin=False)

    parser.add_argument('--add_del_perms', type=str,
                        default='add', help='add/del permissions')

    args = parser.parse_args()

    if args.add_del_perms == 'add':
        cont = 'y'
    elif args.add_del_perms == 'del':
        cont = input('Are you sure you want to remove permission? (y/N): ')
    if cont.lower() != 'y':
        sys.exit(1)

    token, boss_url = get_boss_config(args.boss_config_file)

    meta = BossMeta(args.collection, args.experiment, args.channel)
    rmt = BossRemote(boss_url, token, meta)

    if args.collection not in rmt.list_collections():
        print('collection {} not found'.format(args.collection))
        sys.exit(1)

    if args.group not in rmt.list_groups():
        print('group {} not found'.format(args.group))
        sys.exit(1)

    # Channels have three additional permissions:
    # 'add_volumetric_data'
    # 'read_volumetric_data'
    # 'delete_volumetric_data'

    read_perms = ['read']
    read_vol_perms = ['read_volumetric_data']
    admin_perms = ['add', 'update', 'assign_group', 'remove_group']
    admin_vol_perms = ['add_volumetric_data']
    all_perms = read_perms + admin_perms
    all_vol_perms = read_vol_perms + admin_vol_perms

    if args.add_del_perms == 'add':
        if args.perm_type_admin:
            rmt.add_permissions(args.group, all_perms, all_vol_perms)
        else:
            rmt.add_permissions(args.group, read_perms, read_vol_perms)
    elif args.add_del_perms == 'del':
        if args.perm_type_admin:
            rmt.delete_permissions(args.group, admin_perms, admin_vol_perms)
        else:
            rmt.delete_permissions(args.group, all_perms, all_vol_perms)

    print('Permissions modified')


if __name__ == '__main__':
    main()
