import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

# Κατηγορίες και ετικέτες
category_labels = {
    'accordion': 0,
    'electric_guitar': 1,
    'grand_piano': 2,
    'mandolin': 3,
    'metronome': 4
}
categories = list(category_labels.keys())

# Μεγέθη λεξικού
vocabulary_sizes = [20, 50, 100, 150, 200]

# Τιμές του k
k_values = [1, 3, 5, 7, 9]

# Αποτελέσματα για κάθε μέγεθος λεξικού και k
precision_results = {k: [] for k in k_values}
recall_results = {k: [] for k in k_values}
map_results = {k: [] for k in k_values}

for k in k_values:
    print(f'Testing with k={k}...')
    for size in vocabulary_sizes:
        print(f'Testing with vocabulary size {size} and k={k}...')

        # Φόρτωση λεξικού και SVM
        vocabulary = np.load(f'vocabulary_{size}.npy')
        svm = cv.ml.SVM_create()
        svm = svm.load(f'svm_models/svm_model_{size}_k{k}.xml')

        # Εξαγωγέας περιγραφών BOVW
        descriptor_extractor = cv.BOWImgDescriptorExtractor(cv.xfeatures2d_SIFT.create(), cv.BFMatcher(cv.NORM_L2))
        descriptor_extractor.setVocabulary(vocabulary)

        # Εικόνες δοκιμής και ετικέτες
        test_img_paths = []
        test_labels = []
        for category, label in category_labels.items():
            folder = f'images/caltech-101_5_test/{category}'
            files = [f'{folder}/{file}' for file in os.listdir(folder) if file.endswith('.jpg')]
            test_img_paths.extend(files)
            test_labels.extend([label] * len(files))
        test_labels = np.array(test_labels)

        # Precision και Recall για κάθε κατηγορία
        correct = {label: 0 for label in category_labels.values()}
        retrieved = {label: 0 for label in category_labels.values()}
        relevant = {label: 0 for label in category_labels.values()}

        for img_path, true_label in zip(test_img_paths, test_labels):
            img = cv.imread(img_path)
            kp = cv.xfeatures2d_SIFT.create().detect(img)
            bow_desc = descriptor_extractor.compute(img, kp)

            if bow_desc is None:
                continue

            # Πρόβλεψη SVM
            response = svm.predict(bow_desc)[1]
            predicted_label = int(response[0][0])

            retrieved[predicted_label] += 1
            if predicted_label == true_label:
                correct[predicted_label] += 1
            relevant[true_label] += 1

        # Υπολογισμός Precision και Recall
        precision = {label: correct[label] / retrieved[label] if retrieved[label] > 0 else 0 for label in
                     category_labels.values()}
        recall = {label: correct[label] / relevant[label] if relevant[label] > 0 else 0 for label in
                  category_labels.values()}

        # Υπολογισμός Mean Average Precision (mAP)
        map_value = sum(precision.values()) / len(category_labels)

        # Αποθήκευση αποτελεσμάτων
        precision_results[k].append(np.mean(list(precision.values())))
        recall_results[k].append(np.mean(list(recall.values())))
        map_results[k].append(map_value)

        print(f'Vocabulary size {size}, k={k}: Precision={precision_results[k][-1]:.2f}, Recall={recall_results[k][-1]:.2f}, mAP={map_results[k][-1]:.2f}')

# Σχεδίαση διαγραμμάτων για κάθε k
for k in k_values:
    plt.figure()
    plt.plot(vocabulary_sizes, map_results[k], marker='^', label=f'mAP (k={k})')
    plt.plot(vocabulary_sizes, precision_results[k], marker='o', label=f'Precision (k={k})')
    plt.title(f'Απόδοση για k={k}')
    plt.xlabel('Μέγεθος Λεξικού')
    plt.ylabel('Μετρικές Απόδοσης')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/performance_k{k}.png')
    plt.show()

    # Σχεδίαση διαγράμματος Precision
    plt.figure()
    plt.plot(vocabulary_sizes, recall_results[k], marker='s', label=f'Recall (k={k})')
    plt.title(f'Precision ανά μέγεθος λεξικού για k={k}')
    plt.xlabel('Μέγεθος Λεξικού')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f'plots/precision{k}.png')
    plt.show()



