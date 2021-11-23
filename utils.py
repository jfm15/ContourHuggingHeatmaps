import os
import time
import logging

from config import get_cfg_defaults


def prepare_config_output_and_logger(cfg_path, log_prefix):
    # get config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    # get directory to save log and model
    split_cfg_path = cfg_path.split("/")
    yaml_file_name = os.path.splitext(split_cfg_path[-1])[0]
    output_path = os.path.join('output', yaml_file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(log_prefix, time_str)
    log_path = os.path.join(output_path, log_file)
    save_model_path = os.path.join(output_path, yaml_file_name + "_model.pth")
    save_scaled_model_path = os.path.join(output_path, yaml_file_name + "_scaled_model.pth")

    # setup the logger
    logging.basicConfig(filename=log_path,
                        format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return cfg, logger, output_path, save_model_path, save_scaled_model_path
