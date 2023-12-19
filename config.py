import argparse,json

def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--data_path', type=str, default='../autodl-tmp/ROP_test',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--handcraft_path', type=str, default='./handcraft/zy_shenzhen',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--result_path', type=str, default='./experiments',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--model_dir', type=str, default='./modelCheckPoints',
                        help='Path to the target folder to store the processed datasets.')
    # config file 
    parser.add_argument('--ridge_seg_cfg', help='experiment configuration filename',
                        default="./config_file/hrnet_w48.json", type=str)
    parser.add_argument('--stage_cfg', help='experiment configuration filename',
                        default="./config_file/hrnet_w48.json", type=str)
    args = parser.parse_args()
    # Merge args and config file 
    return args