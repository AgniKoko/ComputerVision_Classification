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

category_labels = {
    'accordion': 0,
    'electric_guitar': 1,
    'grand_piano': 2,
    'mandolin': 3,
    'metronome': 4
}

sift = cv.xfeatures2d_SIFT.create()

for size in vocabulary_sizes:
    print(f'Training for vocabulary size {size}...')

    bow_descs = np.load(f'index/train/train_index_{size}.npy').astype(np.float32)
    img_paths = np.load(f'paths/train/train_paths_{size}.npy', allow_pickle=True)

    labels = []
    for p in img_paths:
        for category, label in category_labels.items():
            if category in p:
                labels.append(label)
                break

    labels = np.array(labels, np.int32)


    print(f'Training SVM for vocabulary size {size}')

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_RBF)
    svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

    svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)

    svm_filename = f'svm_models/svm_model_{size}.xml'
    svm.save(svm_filename)
    print(f'SVM model for vocabulary size {size} saved as {svm_filename}.')

print('SVM training completed for all vocabulary sizes.')
