import numpy as np
import matplotlib.pyplot as plt


# Get the predicted landmark point from the coordinate of the hottest point
def get_hottest_point(heatmap):
    w, h = heatmap.shape
    flattened_heatmap = np.ndarray.flatten(heatmap)
    hottest_idx = np.argmax(flattened_heatmap)
    return np.flip(np.array(np.unravel_index(hottest_idx, [w, h])))


def get_localisation_errors(heatmap_stack, landmarks_per_annotator, pixels_sizes):
    batch_size, no_of_key_points, w, h = heatmap_stack.shape
    radial_error_per_landmark = np.zeros((batch_size, no_of_key_points))

    for i in range(batch_size):

        pixel_size_for_sample = pixels_sizes[i]

        for j in range(no_of_key_points):

            # Get predicted point
            predicted_point = get_hottest_point(heatmap_stack[i, j])
            predicted_point_scaled = predicted_point * pixel_size_for_sample

            # Average the annotators to get target point
            target_point = np.mean(landmarks_per_annotator[i, :, j], axis=0)
            target_point_scaled = target_point * pixel_size_for_sample

            localisation_error = np.linalg.norm(predicted_point_scaled - target_point_scaled)
            radial_error_per_landmark[i, j] = localisation_error

    return radial_error_per_landmark


# This function will visualise the output of the first image in the batch
def visualise(images, heatmap_stack, landmarks_per_annotator):
    s = 0

    # Display image
    image = images[s, 0]
    plt.imshow(image, cmap='gray')

    # Display heatmaps
    normalized_heatmaps = heatmap_stack[s] / np.max(heatmap_stack[s], axis=(1, 2), keepdims=True)
    squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
    plt.imshow(squashed_heatmaps, cmap='inferno', alpha=0.4)

    # Display predicted points
    predicted_landmark_positions = np.array([get_hottest_point(heatmap) for heatmap in normalized_heatmaps])
    plt.scatter(predicted_landmark_positions[:, 0], predicted_landmark_positions[:, 1], color='red', s=2)

    # Display ground truth points
    ground_truth_landmark_position = np.mean(landmarks_per_annotator[s], axis=0)
    plt.scatter(ground_truth_landmark_position[:, 0], ground_truth_landmark_position[:, 1], color='green', s=2)

    plt.show()


