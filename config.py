import argparse,json

def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--data_path', type=str, default='../autodl-tmp/ROP_shen',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--handcraft_path', type=str, default='./handcraft/zy_shenzhen.json',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--result_path', type=str, default='./experiments',
                        help='Path to the target folder to store the processed datasets.')
    parser.add_argument('--model_dir', type=str, default='./modelCheckPoints',
                        help='Path to the target folder to store the processed datasets.')
    
    parser.add_argument('--visual_miss', type=bool, default=False,
                        help='save the result that miss match with the ground truth')
    parser.add_argument('--visual_match', type=bool, default=False,
                        help='save the result that match with the ground truth')
    # config file 
    parser.add_argument('--ridge_seg_cfg', help='experiment configuration filename',
                        default="./configs/ridgeSegmentation/hrnet_w48.json", type=str)
    parser.add_argument('--stage_cfg', help='experiment configuration filename',
                        default="./configs/ROPStage/efficientnet_b7.json", type=str)
    parser.add_argument('--optic_disc_cfg', help='experiment configuration filename',
                        default="./configs/optic_disc/defalut.json", type=str)
    args = parser.parse_args()
    # Merge args and config file 
    return args