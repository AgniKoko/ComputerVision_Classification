import cv2 as cv
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

category_labels = {
    'accordion': 0,
    'electric_guitar': 1,
    'grand_piano': 2,
    'mandolin': 3,
    'metronome': 4
}

sift = cv.xfeatures2d_SIFT.create()

for size in vocabulary_sizes:
    print(f'Training and testing with vocabulary size {size}...')

    for k in k_values:
        print(f'Testing with k={k} for vocabulary size {size}...')

        bow_descs = np.load(f'index/train/train_index_{size}_k{k}.npy').astype(np.float32)
        img_paths = np.load(f'paths/train/train_paths_{size}_k{k}.npy', allow_pickle=True)
        vocabulary = np.load(f'vocabulary_{size}.npy')

        # k-NN train
        labels = []
        for p in img_paths:
            for category, label in category_labels.items():
                if category in p:
                    labels.append(label)
                    break
        labels = np.array(labels, np.int32)

        knn = cv.ml.KNearest_create()
        knn.train(bow_descs, cv.ml.ROW_SAMPLE, labels)

        # BOVW
        descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2))
        descriptor_extractor.setVocabulary(vocabulary)

        # test
        test_img = "images/caltech-101_5_test/grand_piano/image_0080.jpg"
        img = cv.imread(test_img)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        if bow_desc is None:
            print(f'No features detected in test image for vocabulary size {size} and k={k}. Skipping...')
            continue

        # k-NN
        response, results, neighbours, dist = knn.findNearest(bow_desc, k=k)

        # results
        predicted_label = int(response)
        predicted_category = [k for k, v in category_labels.items() if v == predicted_label][0]
        print(f'Vocabulary size {size}, k={k}: It is a {predicted_category}')

print('Testing completed for all vocabulary sizes and k values.')
