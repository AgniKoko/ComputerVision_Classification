import cv2 as cv
import numpy as np

# Νέοι φάκελοι εκπαίδευσης
train_folders = [
    'images/caltech-101_5_train/accordion',
    'images/caltech-101_5_train/electric_guitar',
    'images/caltech-101_5_train/grand_piano',
    'images/caltech-101_5_train/mandolin',
    'images/caltech-101_5_train/metronome'
]

# Μεγέθη λεξικού
vocabulary_sizes = [20, 50, 100, 150, 200]

# Τιμές του k
k_values = [1, 3, 5, 7, 9]

# Κατηγορίες και ετικέτες
category_labels = {
    'accordion': 0,
    'electric_guitar': 1,
    'grand_piano': 2,
    'mandolin': 3,
    'metronome': 4
}

# Αρχικοποίηση SIFT
sift = cv.xfeatures2d_SIFT.create()

for size in vocabulary_sizes:
    print(f'Training for vocabulary size {size}...')

    # Φόρτωση ευρετηρίου και διαδρομών
    bow_descs = np.load(f'index/train/train_index_{size}.npy').astype(np.float32)
    img_paths = np.load(f'paths/train/train_paths_{size}.npy', allow_pickle=True)

    # Ετικέτες εκπαίδευσης
    labels = []
    for p in img_paths:
        for category, label in category_labels.items():
            if category in p:
                labels.append(label)
                break

    labels = np.array(labels, np.int32)

    for k in k_values:
        print(f'Training SVM for vocabulary size {size} and k={k}...')

        # Δημιουργία SVM
        svm = cv.ml.SVM_create()
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setKernel(cv.ml.SVM_RBF)  # Χρήση RBF kernel
        svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

        # Εκπαίδευση SVM
        svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)

        # Αποθήκευση SVM μοντέλου
        svm_filename = f'svm_model/svm_model_{size}_k{k}.xml'
        svm.save(svm_filename)
        print(f'SVM model for vocabulary size {size} and k={k} saved as {svm_filename}.')

print('SVM training completed for all vocabulary sizes and k values.')
