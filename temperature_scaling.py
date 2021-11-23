import argparse
import torch

import model
import numpy as np

from model import two_d_softmax
from model import nll_across_batch
from evaluate import evaluate
from plots import reliability_diagram
from landmark_dataset import LandmarkDataset
from utils import prepare_config_output_and_logger
from torchsummary.torchsummary import summary_string


def parse_args():
    parser = argparse.ArgumentParser(description='Test a network trained to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--fine_tuning_images',
                        help='The path to the images which will be used to fine tune the temperature parameters',
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

    cfg, logger, output_path, _, save_model_path = prepare_config_output_and_logger(args.cfg, 'temperature_scaling')

    # Print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # Print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    # Split the fine-tune set into training and validation
    training_dataset = LandmarkDataset(args.fine_tuning_images, args.annotations, cfg.DATASET,
                                       perform_augmentation=True, subset="first half")
    validation_dataset = LandmarkDataset(args.fine_tuning_images, args.annotations, cfg.DATASET,
                                         perform_augmentation=False, subset="second half")
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1)

    # Load model and state dict from file
    model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS).cuda()
    loaded_state_dict = torch.load(args.pretrained_model)
    model.load_state_dict(loaded_state_dict, strict=True)

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(model, (1, *cfg.DATASET.CACHED_IMAGE_SIZE))
    logger.info(model_summary)

    model.temperatures.requires_grad = True
    optimizer = torch.optim.Adam([model.temperatures], lr=cfg.TRAIN.LR)

    for epoch in range(cfg.TRAIN.EPOCHS):

        logger.info('-----------Epoch {} Temperature Scaling-----------'.format(epoch))

        losses_per_epoch = []

        for batch, (image, channels, meta) in enumerate(training_loader):

            image = image.cuda()
            channels = channels.cuda()

            with torch.no_grad():
                output = model(image.float())

            output = model.scale(output)
            output = two_d_softmax(output)

            optimizer.zero_grad()
            loss = nll_across_batch(output, channels)
            loss.backward()
            optimizer.step()

            losses_per_epoch.append(loss.item())

            if (batch + 1) % 5 == 0:
                logger.info("[{}/{}]\tLoss: {:.3f}".format(batch + 1, len(training_loader), np.mean(losses_per_epoch)))

        msg = "Loss: {:.3f}".format(np.mean(losses_per_epoch))
        logger.info(msg)

        logger.info('-----------Epoch {} Validation-----------'.format(epoch))

        validation_losses = []
        validation_radial_errors = []
        validation_mode_probabilities = []

        with torch.no_grad():
            for idx, (image, channels, meta) in enumerate(validation_loader):

                image = image.cuda()
                channels = channels.cuda()

                output = model(image.float())
                output = model.scale(output)
                output = two_d_softmax(output)

                loss = nll_across_batch(output, channels)
                validation_losses.append(loss.item())

                # Get the radial/localisation error and expected radial error values for each heatmap
                radial_errors, _, mode_probabilities \
                    = evaluate(output.cpu().detach().numpy(),
                               meta['landmarks_per_annotator'].cpu().detach().numpy(),
                               meta['pixel_size'].cpu().detach().numpy())
                validation_radial_errors.append(radial_errors)
                validation_mode_probabilities.append(mode_probabilities)

            flattened_radial_errors = np.array(validation_radial_errors).flatten()
            flattened_mode_probabilities = np.array(validation_mode_probabilities).flatten()

            ece = reliability_diagram(flattened_radial_errors, flattened_mode_probabilities, do_not_save=True)
            logger.info("Loss: {:.3f}\tECE: {:.3f}".format(np.mean(validation_losses), ece))


if __name__ == '__main__':
    main()
