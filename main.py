import os
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

import torch
import torchvision
from torchvision import transforms


# hyper-parameters
# n_clusters_lst = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
n_clusters_lst = [100, 1000, 5000, 10000]


def read_image(data_dir, is_train=False):
    print('Reading MNIST images...')
    transform = transforms.Compose([
                    transforms.ToTensor()
                ])

    image_set = None

    if is_train:
        image_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    else:
        image_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    image_data = []
    for image, target in image_set:
        image_data.append(image.squeeze().numpy())

    return image_data


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def extract_patches(images, size=5):
    print('Extracting 5x5 patches...')
    patches = []
    for image in images:
        image = np.pad(image, 1, pad_with, padder=0.)
        for row_id in range(0, image.shape[0]-size+1, size):
            for column_id in range(0, image.shape[1]-size+1, size):
                patch = image[row_id:row_id+size, column_id:column_id+size]
                patch_flatten = patch.flatten()

                num_nonzeros = np.count_nonzero(patch_flatten)
                if (num_nonzeros / (size * size)) > 0.2:
                    # patch_flatten = preprocessing.normalize(np.expand_dims(patch_flatten, axis=0), norm='l2')
                    # patches.append(np.squeeze(patch_flatten, axis=0))
                    patches.append(patch_flatten)

    return np.array(patches)


def run_kmeans(patches, n_clusters=100):
    print('Run k-means...')
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    kmeans_model.fit(patches)
    return kmeans_model


def plot_cluster_centers(kmeans_model, figsize=10):
    print('Plot cluster centers...')
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    centers = kmeans_model.cluster_centers_.reshape(-1, 5, 5)
    centers_norm = np.zeros((centers.shape[0], centers.shape[1] * centers.shape[2]))

    for id, center in enumerate(centers):
        center_flatten = center.flatten()
        center_norm = preprocessing.normalize(np.expand_dims(center_flatten, axis=0), norm='l2')
        centers_norm[id] = np.squeeze(center_norm, axis=0)

    kmeans_model.cluster_centers_ = centers_norm.astype(np.float32)

    for i, ax in enumerate(ax.flat):
        ax.imshow(centers[i], cmap='gray')
        ax.axis('off')
    plt.savefig(fname=os.path.join(minst_dir, 'kmeans_centers_{}.png'.format(n_clusters)))
    plt.close()


def reconstruct_patches(patches, kmeans_model):
    print('Reconstruct patches...')
    closest_clusters = kmeans_model.predict(patches)
    reconstructed_patches = kmeans_model.cluster_centers_[closest_clusters]
    return reconstructed_patches


def reconstruct_images(images, kmeans_model, size=5):
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(50, 50))

    indices = [0, 1000, 5000, 10000]
    for i, index in enumerate(indices):
        image = images[index]
        image = np.pad(image, 1, pad_with, padder=0.)
        image_recon = np.zeros((30, 30))

        for row_id in range(0, image.shape[0]-size+1, size):
            for column_id in range(0, image.shape[1]-size+1, size):
                patch = image[row_id:row_id+size, column_id:column_id+size]
                patch_flatten = patch.flatten()

                reconstructed_patch = None
                num_nonzeros = np.count_nonzero(patch_flatten)
                if num_nonzeros == 0:
                    reconstructed_patch = patch_flatten
                else:
                    # patch_flatten = preprocessing.normalize(np.expand_dims(patch_flatten, axis=0), norm='l2')
                    # closest_cluster = kmeans_model.predict(patch_flatten)
                    closest_cluster = kmeans_model.predict(np.expand_dims(patch_flatten, axis=0))
                    reconstructed_patch = kmeans_model.cluster_centers_[closest_cluster][0]

                image_recon[row_id:row_id+size, column_id:column_id+size] = reconstructed_patch.reshape((size, size))
        
        ax[i, 0].imshow(image, cmap='gray')
        ax[i, 0].set_title('Original')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(image_recon, cmap='gray')
        ax[i, 1].set_title('Reconstruction')
        ax[i, 1].axis('off')
    
    plt.savefig(fname=os.path.join(minst_dir, 'kmeans_reconstruct_image_{}.png'.format(n_clusters)))
    plt.close()


def plot_reconstruct_patches(original_patches, reconstructed_patches):
    num_patches = original_patches.shape[0]
    indices = np.random.choice(num_patches, 4, replace=False)

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    for i, idx in enumerate(indices):
        original_patch = original_patches[idx].reshape(5, 5)
        reconstructed_patch = reconstructed_patches[idx].reshape(5, 5)
        difference_map = np.abs(original_patch - reconstructed_patch)

        ax[i, 0].imshow(original_patch, cmap='gray')
        ax[i, 0].set_title('Original')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(reconstructed_patch, cmap='gray')
        ax[i, 1].set_title('Reconstructed')
        ax[i, 1].axis('off')

        ax[i, 2].imshow(difference_map, cmap='hot')
        ax[i, 2].set_title('Difference')
        ax[i, 2].axis('off')
    
    plt.savefig(fname=os.path.join(minst_dir, 'kmeans_comparison_{}.png'.format(n_clusters)))
    plt.close()


def calculate_patch_error(original_patches, reconstructed_patches):
    mse = np.mean((original_patches - reconstructed_patches)**2)
    return mse


def plot_mse_curve(mse_loss):
    indices = n_clusters_lst

    # Create the plot
    plt.plot(indices, mse_loss)

    # Customize the plot
    plt.xlabel('Number of clusters')
    plt.ylabel('MSE')
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.savefig(fname=os.path.join(minst_dir, 'kmeans_reconstruction_mse.png'))
    plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform kmeans')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the MNIST dataset')
    args = parser.parse_args()

    # Data directory
    minst_dir = args.image_dir

    train_data = read_image(minst_dir, is_train=True)
    train_patches = extract_patches(train_data, size=5)

    mse_list = []
    for n_clusters in n_clusters_lst:
        figsize = int(math.sqrt(n_clusters))
        kmeans_model = run_kmeans(train_patches, n_clusters=n_clusters)
        reconstruct_images(train_data, kmeans_model, size=5)
        plot_cluster_centers(kmeans_model, figsize=figsize)
        reconstructed_patches = reconstruct_patches(train_patches, kmeans_model)
        plot_reconstruct_patches(train_patches, reconstructed_patches)
        mse = calculate_patch_error(train_patches, reconstructed_patches)
        mse_list.append(mse)

    plot_mse_curve(mse_list)
