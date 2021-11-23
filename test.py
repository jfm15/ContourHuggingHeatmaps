import argparse
import torch

import model

from model import two_d_softmax
from model import nll_across_batch
from evaluate import get_localisation_errors
from evaluate import visualise
from landmark_dataset import LandmarkDataset
from utils import prepare_config_output_and_logger
from torchsummary.torchsummary import summary_string


def parse_args():
    parser = argparse.ArgumentParser(description='Test a network trained to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--testing_images',
                        help='The path to the testing images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--pretrained_model',
                        help='the path to a pretrained model',
                        type=str,
                        required=True)

    args = parser.parse_args()

    return args


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg, logger, _ = prepare_config_output_and_logger(args.cfg)

    # Print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # Print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    # Load the testing dataset and put it into a loader
    test_dataset = LandmarkDataset(args.testing_images, args.annotations, cfg.DATASET, perform_augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model and state dict from file
    model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS)
    loaded_state_dict = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
    model.load_state_dict(loaded_state_dict, strict=True)

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(model, (1, *cfg.DATASET.CACHED_IMAGE_SIZE), device=torch.device('cpu'))
    logger.info(model_summary)

    logger.info("-----------Start Testing-----------")
    model.eval()
    all_losses = []

    with torch.no_grad():
        for idx, (image, channels, meta) in enumerate(test_loader):

            output = model(image.float())
            output = model.scale(output)
            output = two_d_softmax(output)

            loss = nll_across_batch(output, channels)
            all_losses.append(loss.item())

            # Get the radial/localisation error and expected radial error values for each heatmap
            radial_errors = get_localisation_errors(output.detach().numpy(),
                                                    meta['landmarks_per_annotator'].detach().numpy(),
                                                    meta['pixel_size'].detach().numpy())
            print(radial_errors)

            visualise(image.cpu().detach().numpy(),
                      output.cpu().detach().numpy(),
                      meta['landmarks_per_annotator'].cpu().detach().numpy())

            # Print loss, radial error for each landmark and MRE for the image
            msg = "Image: {}\tLoss: {:.3f}".format(meta['file_name'][0], loss.item())
            logger.info(msg)


if __name__ == '__main__':
    main()