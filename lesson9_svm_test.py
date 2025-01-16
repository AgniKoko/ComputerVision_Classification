import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

category_labels = {
    'accordion': 0,
    'electric_guitar': 1,
    'grand_piano': 2,
    'mandolin': 3,
    'metronome': 4
}
categories = list(category_labels.keys())

vocabulary_sizes = [20, 50, 100, 150, 200]

precision_results = []
recall_results = []
map_results = []


for size in vocabulary_sizes:
    print(f'Testing with vocabulary size {size}')

    # SVM
    vocabulary = np.load(f'vocabulary_{size}.npy')
    svm = cv.ml.SVM_create()
    svm = svm.load(f'svm_models/svm_model_{size}.xml')

    # BOVW
    descriptor_extractor = cv.BOWImgDescriptorExtractor(cv.xfeatures2d_SIFT.create(), cv.BFMatcher(cv.NORM_L2))
    descriptor_extractor.setVocabulary(vocabulary)

    test_img_paths = []
    test_labels = []
    for category, label in category_labels.items():
        folder = f'images/caltech-101_5_test/{category}'
        files = [f'{folder}/{file}' for file in os.listdir(folder) if file.endswith('.jpg')]
        test_img_paths.extend(files)
        test_labels.extend([label] * len(files))
    test_labels = np.array(test_labels)

    correct = {label: 0 for label in category_labels.values()}
    retrieved = {label: 0 for label in category_labels.values()}
    relevant = {label: 0 for label in category_labels.values()}

    for img_path, true_label in zip(test_img_paths, test_labels):
        img = cv.imread(img_path)
        kp = cv.xfeatures2d_SIFT.create().detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        if bow_desc is None:
            continue

        response = svm.predict(bow_desc)[1]
        predicted_label = int(response[0][0])

        retrieved[predicted_label] += 1
        if predicted_label == true_label:
            correct[predicted_label] += 1
        relevant[true_label] += 1

    # Precision Recall
    precision = {label: correct[label] / retrieved[label] if retrieved[label] > 0 else 0 for label in
                 category_labels.values()}
    recall = {label: correct[label] / relevant[label] if relevant[label] > 0 else 0 for label in
              category_labels.values()}

    # Mean Average Precision (mAP)
    map_value = sum(precision.values()) / len(category_labels)

    precision_results.append(np.mean(list(precision)))
    recall_results.append(np.mean(list(recall)))
    map_results.append(map_value)

    print(f'Vocabulary size {size}, Precision={precision_results[-1]:.2f}, Recall={recall_results[-1]:.2f}, mAP={map_results[-1]:.2f}')


plt.figure()
plt.plot(vocabulary_sizes, map_results, marker='^', label=f'mAP')
plt.plot(vocabulary_sizes, precision_results, marker='o', label=f'Precision')
plt.title(f'Απόδοση')
plt.xlabel('Μέγεθος Λεξικού')
plt.ylabel('Μετρικές Απόδοσης')
plt.legend()
plt.grid(True)
plt.savefig(f'plots/performance.png')
plt.show()

# Precision
plt.figure()
plt.plot(vocabulary_sizes, recall_results, marker='s', label=f'Recall')
plt.title(f'Precision ανά μέγεθος λεξικού')
plt.xlabel('Μέγεθος Λεξικού')
plt.ylabel('Precision')
plt.grid(True)
plt.savefig(f'plots/precision.png')
plt.show()



