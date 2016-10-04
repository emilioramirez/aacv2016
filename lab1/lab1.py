# -*- coding: utf-8 -*-
'''
AAVC, FaMAF-UNC, 20-SEP-2016

=======================================================
Lab 1: Clasificación de imágenes empleando modelos BoVW
=======================================================

0) Familiarizarse con el código y con el dataset.

Svetlana Lazebnik, Cordelia Schmid, and Jean Ponce. (2006) Beyond Bags of
Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories. In:
CVPR. Link:
http://www-cvr.ai.uiuc.edu/ponce_grp/data/scene_categories/scene_categories.zip

1) Determinar el mejor valor para el parámetro C del clasificador (SVM lineal)
mediante 5-fold cross-validation en el conjunto de entrenamiento (gráfica) de
uno de los folds. Una vez elegido el parámetro, reportar media y desviación
estándar del accuracy sobre el conjunto de test.


Hsu, C. W., Chang, C. C., & Lin, C. J. (2003). A practical guide to support
vector classification.

2) Evaluar accuracy vs. n_clusters. Que pasa con los tiempos de cómputo de
BoVW?. Gráficas.

50 - 90 - 140

Si tengo descriptores locales L2-normalizados: cómo puedo optimizar la
asignación de los mismos a palabras del diccionario? (ayuda: expresar la
distancia euclídea entre dos vectores en términos de productos puntos entre los
mismo)

3) Transformaciones en descriptores / vectores BoVW: evaluar impacto de
transformación sqrt y norma L2.

Arandjelović, R., & Zisserman, A. (2012). Three things everyone should know to
improve object retrieval. In: CVPR.

4) Kernels no lineales: Intersección (BoVW: norm=1) y RBF, ajustando parámetros
mediante cross-validation en conjunto de validación.

5*) Implementar "spatial augmentation": agregar las coordenadas espaciales
(relativas) a cada descriptor, esto es: el descriptor d=(d1,...,dn) se
transforma a d'=(d1,...,dn, x/W-0.5, y/H-0.5), en donde H y W son el alto y
ancho de la imagen, respectivamente.

Sánchez, J., Perronnin, F., & De Campos, T. (2012). Modeling the spatial layout
of images beyond spatial pyramids. In: PRL

6*) Emplear un "vocabulary tree". Explicar como afecta la asignación de
descritpores locales a palabras del diccionario.

Nister, D., & Stewenius, H. (2006). Scalable recognition with a vocabulary
tree. In: CVPR

7*) Reemplazar BoVW por VLAD (implementar)

Arandjelovic, R., & Zisserman, A. (2013). All about VLAD. In: CVPR

8*) Trabajar sobre el dataset MIT-IndoorScenes (67 clases).

A. Quattoni, and A.Torralba (2009). Recognizing Indoor Scenes. In: CVPR.
link: http://web.mit.edu/torralba/www/indoor.html

Algunas observaciones:

  - El dataset provee un train/test split estándar, por lo que hay que armar un
  parser que levante los .txt y arme el diccionario correspondiente al
  dataset.

  - Son 2.4G de imágenes, por lo que tener todos los vectores BoVW en memoria se
  vuelve difícil. El entrenamiento en este caso se debe hacer mediante SGD
  (sklearn.linear_models.SGDClassifier). Prestar atención al esquema de
  actualización.
'''
from __future__ import print_function
from __future__ import division

import sys

from os import listdir, makedirs
from os.path import join, splitext, abspath, split, exists

import numpy as np
np.seterr(all='raise')

import cv2

from skimage.feature import daisy
from skimage.transform import rescale

from sklearn.svm import LinearSVC, SVC

from scipy.spatial import distance
from scipy.io import savemat, loadmat

DAISY_L2 = '-daisy:L2'
DAISY_sqrt = '-daisy:sqrt'
DAISY_L2_sqrt = '-daisy:L2+sqrt'
DAISY_NONE = '-daisy:None'
DAISY_L1 = '-daisy:L1'
EXT_FEAT = '{}.feat'#'.feat'

VOCABYLARY = 'vocabulary{:d}{}.dat'
SAMPLE = 'sample{:d}{}.feat'

CURRENT_DAISY = DAISY_L1


BOVW_L2 = '-bovw:L2'
BOVW_sqrt = '-bovw:sqrt'
BOVW_L2_sqrt = '-bovw:L2+sqrt'
BOVW_None = '-bovw:None'
BOVW_L1 = '-bovw:L2'  # Dice L2 pero en realidad es L1


CURRENT_BOVW = BOVW_L1

EXT_BOVW = '{}{}.bovw'


def save_data(data, filename, force_overwrite=False):
    # if dir/subdir doesn't exist, create it
    dirname = split(filename)[0]
    if not exists(dirname):
        makedirs(dirname)
    savemat(filename, {'data': data}, appendmat=False)


def load_data(filename):
    return loadmat(filename, appendmat=False)['data'].squeeze()


def load_scene_categories(path, random_state=None):
    cname = sorted(listdir(path))  # human-readable names
    cid = []                       # class id wrt cname list
    fname = []                     # relative file paths
    for i, cls in enumerate(cname):
        for img in listdir(join(path, cls)):
            if splitext(img)[1] not in ('.jpeg', '.jpg', '.png'):
                continue
            fname.append(join(cls, img))
            cid.append(i)
    return {'cname': cname, 'cid': cid, 'fname': fname}


def n_per_class_split(dataset, n=100, random_state=None):
    # set RNG
    if random_state is None:
        random_state = np.random.RandomState()

    n_classes = len(dataset['cname'])
    cid = dataset['cid']
    fname = dataset['fname']

    train_set = []
    test_set = []
    for id_ in range(n_classes):
        idxs = [i for i, j in enumerate(cid) if j == id_]
        random_state.shuffle(idxs)

        # train samples
        for i in idxs[:n]:
            train_set.append((fname[i], cid[i]))

        # test samples
        for i in idxs[n:]:
            test_set.append((fname[i], cid[i]))

    random_state.shuffle(train_set)
    random_state.shuffle(test_set)
    return train_set, test_set


SCALES_3 = [1.0, 0.5, 0.25]
SCALES_5 = [1.0, 0.707, 0.5, 0.354, 0.25]
def extract_multiscale_dense_features(imfile, step=8, scales=SCALES_3):
    im = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
    feat_all = []
    for sc in scales:
        dsize = (int(sc * im.shape[0]), int(sc * im.shape[1]))
        im_scaled = cv2.resize(im, dsize, interpolation=cv2.INTER_LINEAR)
        feat = daisy(im_scaled, step=step)
        if feat.size == 0:
            break
        ndim = feat.shape[2]
        feat = np.atleast_2d(feat.reshape(-1, ndim))
        feat_all.append(feat)
    return np.row_stack(feat_all).astype(np.float32)


def compute_features(base_path, im_list, output_path):
    # compute and store low level features for all images
    for fname in im_list:
        # image full path
        imfile = join(base_path, fname)

        # check if destination file already exists
        featfile = join(output_path, splitext(fname)[0] + EXT_FEAT.format(CURRENT_DAISY))
        current_file = featfile
        if exists(current_file):
            print('{} already exists'.format(current_file))
            continue

        feat = extract_multiscale_dense_features(imfile)
        save_data(feat, featfile)

        # feat_l2 = join(output_path, splitext(fname)[0] + EXT_FEAT.format(CURRENT_DAISY))
        # current_file = feat_l2
        # save_data(normalize_L2(feat), feat_l2)

        # feat_sqrt = join(output_path, splitext(fname)[0] + EXT_FEAT.format(CURRENT_DAISY))
        # current_file = feat_sqrt
        # save_data(normalize_sqrt(feat), feat_sqrt)

        # feat_l2_sqrt = join(output_path, splitext(fname)[0] + EXT_FEAT.format(CURRENT_DAISY))
        # current_file = feat_l2_sqrt
        # save_data(normalize_L2(normalize_sqrt(feat)), feat_l2_sqrt)

        print('{}: {} features'.format(featfile, feat.shape[0]))


def normalize_L2(vector, norm=2):
    nrm = np.linalg.norm(vector, ord=norm)
    return vector / (nrm + 1e-7)


def normalize_sqrt(vector):
    return np.sign(vector) * np.sqrt(np.absolute(vector))


def sample_feature_set(base_path, im_list, output_path, n_samples, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    n_per_file = 100
    sample_file = join(output_path, SAMPLE.format(n_samples, CURRENT_DAISY))
    if exists(sample_file):
        sample = load_data(sample_file)
    else:
        sample = []
        while len(sample) < n_samples:
            i = random_state.randint(0, len(im_list))
            featfile = join(base_path, splitext(im_list[i])[0] + EXT_FEAT.format(CURRENT_DAISY))
            feat = load_data(featfile)
            idxs = random_state.choice(range(feat.shape[0]), 100)
            sample += [feat[i] for i in idxs]
            print('\r{}/{} samples'.format(len(sample), n_samples), end='')
            sys.stdout.flush()

        sample = np.row_stack(sample)
        save_data(sample, sample_file)
    print('\r{}: {} features'.format(sample_file, sample.shape[0]))
    return sample


def kmeans_fit(samples, n_clusters, maxiter=100, tol=1e-4, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    n_samples = samples.shape[0]

    # chose random samples as initial estimates
    idxs = random_state.randint(0, n_samples, n_clusters)
    centroids = samples[idxs, :]

    J_old = np.inf
    for iter_ in range(maxiter):

        # SAMPLE-TO-CLUSTER ASSIGNMENT

        # cdist returns a matrix of size n_samples x n_clusters, where the i-th
        # row stores the (squared) distance from sample i to all centroids
        dist2 = distance.cdist(samples, centroids, metric='sqeuclidean')

        # argmin over columns of the distance matrix
        assignment = np.argmin(dist2, axis=1)

        # CENTROIDS UPDATE (+ EVAL DISTORTION)

        J_new = 0.
        for k in range(n_clusters):
            idxs = np.where(assignment == k)[0]
            if len(idxs) == 0:
                raise RuntimeError('k-means crash!')

            centroids[k, :] = np.mean(samples[idxs], axis=0).astype(np.float32)

            J_new += np.sum(dist2[idxs, assignment[idxs]])
        J_new /= float(n_samples)

        print('iteration {}, potential={:.3e}'.format(iter_, J_new))
        if (J_old - J_new) / J_new < tol:
            print('STOP')
            break
        J_old = J_new

    return centroids


def compute_bovw(vocabulary, features, norm=2):
    if vocabulary.shape[1] != features.shape[1]:
        raise RuntimeError('something is wrong with the data dimensionality')
    dist2 = distance.cdist(features, vocabulary, metric='sqeuclidean')
    assignments = np.argmin(dist2, axis=1)
    bovw, _ = np.histogram(assignments, range(vocabulary.shape[1]))
    return bovw


def split_into_X_y(dataset, output_path='cache'):
    X, y = [], []
    for fname, cid in dataset:
        bovwfile = join(output_path, splitext(fname)[0] + EXT_BOVW.format(CURRENT_DAISY, CURRENT_BOVW))
        #X.append(pickle.load(open(bovwfile, 'rb')))
        X.append(load_data(bovwfile))
        y.append(cid)
    return (np.array(X), np.array(y))


if __name__ == "__main__":
    random_state = np.random.RandomState(12345)

    # ----------------
    # DATA PREPARATION
    # ----------------

    N_CLUSTER = 100
    # paths
    dataset_path = abspath('scene_categories')
    output_path = 'cache'

    # load dataset
    dataset = load_scene_categories(dataset_path)
    n_classes = len(dataset['cname'])
    n_images = len(dataset['fname'])
    print('{} images of {} categories'.format(n_images, n_classes))

    # train-test split
    train_set, test_set = n_per_class_split(dataset, n=100)
    n_train = len(train_set)
    n_test = len(test_set)
    print('{} training samples / {} testing samples'.format(n_train, n_test))

    # compute and store low level features for all images
    # compute_features(dataset_path, dataset['fname'], output_path)

    # --------------------------------
    # UNSUPERVISED DICTIONARY LEARNING
    # --------------------------------

    # n_samples = int(1e5)
    # n_clusters = N_CLUSTER
    # vocabulary_file = join(output_path, VOCABYLARY.format(n_clusters, CURRENT_DAISY))
    # if exists(vocabulary_file):
    #     #vocabulary = pickle.load(open(vocabulary_file, 'rb'))
    #     vocabulary = load_data(vocabulary_file)
    # else:
    #     train_files = [fname for (fname, cid) in train_set]
    #     sample = sample_feature_set(output_path, train_files, output_path,
    #                                 n_samples, random_state=random_state)
    #     vocabulary = kmeans_fit(sample, n_clusters=n_clusters,
    #                             random_state=random_state)
    #     save_data(vocabulary, vocabulary_file)

    # print('{}: {} clusters'.format(vocabulary_file, vocabulary.shape[0]))

    # --------------------
    # COMPUTE BoVW VECTORS
    # --------------------
    # from datetime import datetime
    # start_time = datetime.now()
    # for fname in dataset['fname']:
    #     # low-level features file
    #     featfile = join(output_path, splitext(fname)[0] + EXT_FEAT.format(CURRENT_DAISY))

    #     # check if destination file already exists
    #     bovwfile = join(output_path, splitext(fname)[0] + EXT_BOVW.format(CURRENT_DAISY, CURRENT_BOVW))
    #     if exists(bovwfile):
    #         print('{} already exists'.format(bovwfile))
    #         continue

    #     #feat = pickle.load(open(featfile, 'rb'))
    #     feat = load_data(featfile)
    #     bovw = compute_bovw(vocabulary, feat)

    #     #save_data(bovw, bovwfile)
    #     # print("NO SE GRABO LA NORMALIZACION CORRECTA DEL CURRENT_BOVW")

    #     # bovwfile_l2 = join(output_path, splitext(fname)[0] + EXT_BOVW.format(CURRENT_DAISY, BOVW_L2))
    #     # bovw_l2 = normalize_L2(bovw)
    #     # save_data(bovw_l2, bovwfile_l2)

    #     # bovwfile_l1 = join(output_path, splitext(fname)[0] + EXT_BOVW.format(CURRENT_DAISY, BOVW_L2))
    #     # bovw_l1 = normalize_L2(bovw, norm=1)
    #     # save_data(bovw_l1, bovwfile_l1)

    #     # bovwfile_sqrt = join(output_path, splitext(fname)[0] + EXT_BOVW.format(CURRENT_DAISY, BOVW_sqrt))
    #     # bovw_sqrt = normalize_sqrt(bovw)
    #     # save_data(bovw_sqrt, bovwfile_sqrt)

    #     # bovwfile_l2_sqrt = join(output_path, splitext(fname)[0] + EXT_BOVW.format(CURRENT_DAISY, BOVW_L2_sqrt))
    #     # bovw_l2_sqrt = normalize_L2(normalize_sqrt(bovw))
    #     # save_data(bovw_l2_sqrt, bovwfile_l2_sqrt)

    #     print('{}'.format(bovwfile))
    # stop_time = datetime.now()
    # time_lapse = stop_time - start_time
    # print("time lapse on bovw for {} clusters:".format(n_clusters), time_lapse.total_seconds())

    # -----------------
    # TRAIN CLASSIFIERS
    # -----------------

    # setup training data
    X_train, y_train = split_into_X_y(train_set)

    # svm = LinearSVC(C=10.0)
    # # svm = SVC(C=10.0, gamma=10)
    # svm.fit(X_train, y_train)

    # # setup testing data
    # X_test, y_test = split_into_X_y(test_set)

    # y_pred = svm.predict(X_test)

    # tp = np.sum(y_test == y_pred)
    # print('accuracy = {:.3f}'.format(float(tp) / len(y_test)))

    from utils import cross_validation
    print(cross_validation(X_train, y_train))
