import cv2 as cv
import os
import numpy as np

train_folders = [
    'images/caltech-101_5_train/accordion',
    'images/caltech-101_5_train/electric_guitar',
    'images/caltech-101_5_train/grand_piano',
    'images/caltech-101_5_train/mandolin',
    'images/caltech-101_5_train/metronome'
]

vocabulary_sizes = [20, 50, 100, 150, 200]
k_values = [1, 3, 5, 7, 9]

sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

for size in vocabulary_sizes:
    print(f'Processing vocabulary size {size}...')

    # Extract Database
    print('Extracting features...')
    train_descs = np.zeros((0, 128))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            desc = extract_local_features(path)
            if desc is None:
                continue
            train_descs = np.concatenate((train_descs, desc), axis=0)

    # Create vocabulary
    print(f'Creating vocabulary of size {size}...')
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv.BOWKMeansTrainer(size, term_crit, 1, cv.KMEANS_PP_CENTERS)
    vocabulary = trainer.cluster(train_descs.astype(np.float32))

    np.save(f'vocabulary_{size}.npy', vocabulary)

    # Create index
    print('Creating index for vocabulary size {size}...')
    descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2))
    descriptor_extractor.setVocabulary(vocabulary)

    img_paths = []
    bow_descs = np.zeros((0, vocabulary.shape[0]))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)

            img = cv.imread(path)
            kp = sift.detect(img)
            bow_desc = descriptor_extractor.compute(img, kp)

            if bow_desc is None:
                continue

            img_paths.append(path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

    for k in k_values:
        np.save(f'index/train/train_index_{size}_k{k}.npy', bow_descs)
        np.save(f'paths/train/train_paths_{size}_k{k}.npy', img_paths)

    print(f'Vocabulary, index, and paths saved for vocabulary size {size} and k values {k_values}.')
