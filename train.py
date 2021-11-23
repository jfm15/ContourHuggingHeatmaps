import argparse
import torch

import model
import numpy as np
import matplotlib.pyplot as plt

from model import two_d_softmax
from model import nll_across_batch
from config import get_cfg_defaults
from landmark_dataset import LandmarkDataset
from utils import prepare_config_output_and_logger
from torchsummary.torchsummary import summary_string


'''
Code design based on Bin Xiao's Deep High Resolution Network Repository:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network to detect landmarks')

    parser.add_argument('--cfg',
                        help='The path to the configuration file for the experiment',
                        required=True,
                        type=str)

    parser.add_argument('--training_images',
                        help='The path to the training images',
                        type=str,
                        required=True,
                        default='')

    parser.add_argument('--annotations',
                        help='The path to the directory where annotations are stored',
                        type=str,
                        required=True,
                        default='')

    args = parser.parse_args()

    return args


def main():
    # get arguments and the experiment file
    args = parse_args()

    cfg, logger, save_model_path = prepare_config_output_and_logger(args.cfg)

    # print the arguments into the log
    logger.info("-----------Arguments-----------")
    logger.info(vars(args))
    logger.info("")

    # print the configuration into the log
    logger.info("-----------Configuration-----------")
    logger.info(cfg)
    logger.info("")

    # load the train dataset and put it into a loader
    train_dataset = LandmarkDataset(args.training_images, args.annotations, cfg.DATASET, perform_augmentation=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    '''
    for batch, (image, channels, meta) in enumerate(train_loader):
        s = 0
        plt.imshow(image[s].detach().numpy(), cmap='gray')
        squashed_channels = np.max(channels[s].detach().numpy(), axis=0)
        plt.imshow(squashed_channels, cmap='inferno', alpha=0.5)

        landmarks_per_annotator = meta['landmarks_per_annotator'].detach().numpy()[s]
        averaged_landmarks = np.mean(landmarks_per_annotator, axis=0)
        for i, position in enumerate(averaged_landmarks):
            plt.text(position[0], position[1], "{}".format(i + 1), color="yellow", fontsize="small")
        plt.show()
    '''

    model = eval("model." + cfg.MODEL.NAME)(cfg.MODEL, cfg.DATASET.KEY_POINTS).cuda()

    logger.info("-----------Model Summary-----------")
    model_summary, _ = summary_string(model, (1, *cfg.DATASET.CACHED_IMAGE_SIZE))
    logger.info(model_summary)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 6, 8], gamma=0.1)

    for epoch in range(cfg.TRAIN.EPOCHS):

        logger.info('-----------Epoch {} Training-----------'.format(epoch))

        model.train()
        losses_per_epoch = []

        for batch, (image, channels, meta) in enumerate(train_loader):

            # Put image and channels onto gpu
            image = image.cuda()
            channels = channels.cuda()

            output = model(image.float())
            output = two_d_softmax(output)

            optimizer.zero_grad()
            loss = nll_across_batch(output, channels)
            loss.backward()

            optimizer.step()

            losses_per_epoch.append(loss.item())

            if (batch + 1) % 5 == 0:
                logger.info("[{}/{}]\tLoss: {:.3f}".format(batch + 1, len(train_loader), np.mean(losses_per_epoch)))

        scheduler.step()


if __name__ == '__main__':
    main()