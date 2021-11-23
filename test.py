import os
import argparse
import torch

import model
import numpy as np

from model import two_d_softmax
from model import nll_across_batch
from evaluate import evaluate
from evaluate import visualise
from evaluate import produce_sdr_statistics
from plots import radial_error_vs_ere_graph
from plots import roc_outlier_graph
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

    parser.add_argument('--outlier_threshold',
                        help='Classify landmarks with an ERE score over this value as outliers',
                        type=float,
                        default=1.5)

    args = parser.parse_args()

    return args


def main():

    # Get arguments and the experiment file
    args = parse_args()

    cfg, logger, output_path, _, _ = prepare_config_output_and_logger(args.cfg, 'test')

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
    all_radial_errors = []
    all_expected_radial_errors = []
    all_mode_probabilities = []

    with torch.no_grad():
        for idx, (image, channels, meta) in enumerate(test_loader):

            output = model(image.float())
            output = model.scale(output)
            output = two_d_softmax(output)

            loss = nll_across_batch(output, channels)
            all_losses.append(loss.item())

            # Get the radial/localisation error and expected radial error values for each heatmap
            radial_errors, expected_radial_errors, mode_probabilities\
                = evaluate(output.detach().numpy(),
                           meta['landmarks_per_annotator'].detach().numpy(),
                           meta['pixel_size'].detach().numpy())
            all_radial_errors.append(radial_errors)
            all_expected_radial_errors.append(expected_radial_errors)
            all_mode_probabilities.append(mode_probabilities)

            # Print loss, radial error for each landmark and MRE for the image
            # Assumes that the batch size is 1 here
            msg = "Image: {}\tloss: {:.3f}".format(meta['file_name'][0], loss.item())
            for radial_error in radial_errors[0]:
                msg += "\t{:.3f}mm".format(radial_error)
            msg += "\taverage: {:.3f}mm".format(np.mean(radial_errors))
            logger.info(msg)

            '''
            visualise(image.cpu().detach().numpy(),
                      output.cpu().detach().numpy(),
                      meta['landmarks_per_annotator'].cpu().detach().numpy())
            '''

    # Print out the statistics and graphs shown in the paper
    logger.info("\n-----------Final Statistics-----------")

    # Overall loss
    logger.info("Average loss: {:.3f}".format(np.mean(all_losses)))

    # MRE per landmark
    all_radial_errors = np.array(all_radial_errors)
    mre_per_landmark = np.mean(all_radial_errors, axis=(0, 1))
    msg = "Average radial error per landmark: "
    for mre in mre_per_landmark:
        msg += "\t{:.3f}mm".format(mre)
    logger.info(msg)

    # Total MRE
    mre = np.mean(all_radial_errors)
    logger.info("Average radial error (MRE): {:.3f}mm".format(mre))

    # Detection rates
    flattened_radial_errors = all_radial_errors.flatten()
    sdr_statistics = produce_sdr_statistics(flattened_radial_errors, [2.0, 2.5, 3.0, 4.0])
    logger.info("Successful Detection Rate (SDR) for 2mm, 2.5mm, 3mm and 4mm respectively: "
                "{:.3f}% {:.3f}% {:.3f}% {:.3f}%".format(*sdr_statistics))

    # Generate graphs
    logger.info("\n-----------Save Graphs-----------")
    flattened_expected_radial_errors = np.array(all_expected_radial_errors).flatten()
    all_mode_probabilities = np.array(all_mode_probabilities).flatten()

    # Save the correlation between radial error and ere graph
    graph_save_path = os.path.join(output_path, "re_vs_ere_correlation_graph")
    logger.info("Saving radial error vs expected radial error (ERE) graph to => {}".format(graph_save_path))
    radial_error_vs_ere_graph(flattened_radial_errors, flattened_expected_radial_errors, graph_save_path)

    # Save the roc outlier graph
    graph_save_path = os.path.join(output_path, "roc_outlier_graph")
    logger.info("Saving roc outlier graph to => {}".format(graph_save_path))
    proposed_threshold = roc_outlier_graph(flattened_radial_errors, flattened_expected_radial_errors, graph_save_path)

    # Save the roc outlier graph
    graph_save_path = os.path.join(output_path, "reliability_diagram")
    logger.info("Saving reliability diagram to => {}".format(graph_save_path))
    reliability_diagram(flattened_radial_errors, all_mode_probabilities, graph_save_path)

    logger.info("\n-----------Outlier Prediction Experiment-----------")

    # Outlier threshold proposal
    logger.info("Classifying heatmaps with an ERE > {:.3f} produces "
                "a true positive rate of 0.5 for detecting outliers".format(proposed_threshold))

    logger.info("Using {:.3f} as a threshold for ERE we split the overall "
                "set into a 'good' and 'erroneous' set with the following statistics:".format(args.outlier_threshold))
    good_set_radial_errors = flattened_radial_errors[flattened_radial_errors <= args.outlier_threshold]
    good_set_sdr_statistics = produce_sdr_statistics(good_set_radial_errors, [2.0, 2.5, 3.0, 4.0])
    logger.info("A good set with {} landmarks for which the MRE is {:.3f}mm and the (SDR) "
                "for 2mm, 2.5mm, 3mm and 4mm respectively is {:.3f}% {:.3f}% {:.3f}% {:.3f}%"
                .format(len(good_set_radial_errors), np.mean(good_set_radial_errors), *good_set_sdr_statistics))

    erroneous_set_radial_errors = flattened_radial_errors[flattened_radial_errors > args.outlier_threshold]
    erroneous_set_sdr_statistics = produce_sdr_statistics(erroneous_set_radial_errors, [2.0, 2.5, 3.0, 4.0])
    logger.info("An erroneous set with {} landmarks for which the MRE is {:.3f}mm and the (SDR) "
                "for 2mm, 2.5mm, 3mm and 4mm respectively is {:.3f}% {:.3f}% {:.3f}% {:.3f}%"
                .format(len(erroneous_set_radial_errors), np.mean(erroneous_set_radial_errors),
                        *erroneous_set_sdr_statistics))





if __name__ == '__main__':
    main()