<h1>atelier 2 Deep learning</h1>
<H3>1. Establish a CNN Architecture (Based on Pytorch Library) to classify MINST Dataset, by
defining layers (Convolution, pooling, fully connect layer), the hyper-parameters (Kernels,
Padding , stride, optimizers, regularization, etc) and running the model in GPU mode.</H3>
<p>J'ai chargé les données MNIST en créant deux fonctions spécifiques :

load_mnist_images(filename) : Cette fonction charge les images à partir du fichier binaire, les normalise entre 0 et 1, puis les retourne sous forme de tableau numpy.
load_mnist_labels(filename) : Cette fonction charge les labels associés aux images (les chiffres de 0 à 9) à partir du fichier binaire.
J'ai ensuite défini les chemins vers les fichiers MNIST et utilisé ces fonctions pour charger les images et les labels des ensembles d'entraînement et de test.

J'ai préparé les données pour PyTorch en convertissant les images et les labels en tensors PyTorch. Pour les images, j'ai ajouté une dimension supplémentaire pour le canal (gris, donc une seule dimension). Ensuite, j'ai créé des Datasets PyTorch pour l'entraînement et les tests à partir des images et des labels. J'ai également créé des DataLoaders pour charger les données en batchs pendant l'entraînement et les tests.

J'ai défini un modèle de réseau de neurones convolutionnel (CNN) en créant une classe CNNModel qui hérite de nn.Module. Voici les principales couches du modèle :

Couches convolutionnelles :
conv1 : 32 filtres, noyau de 3x3, stride de 1, padding de 1.
conv2 : 64 filtres.
conv3 : 128 filtres.
Couches de pooling :
J'ai utilisé une couche MaxPool2d pour réduire la taille des images de moitié à chaque étape.
Couches entièrement connectées (fully connected) :
Après les convolutions, j'ai aplati les données et les ai envoyées dans des couches fully connected pour la classification des chiffres.
J'ai ensuite défini une fonction train_model pour entraîner le modèle. Voici les étapes de l'entraînement :

Le modèle est déplacé sur le GPU si disponible.
À chaque époque, les images et les labels sont extraits du DataLoader, envoyés sur le GPU, et traités dans le modèle.
J'ai utilisé CrossEntropyLoss comme fonction de perte, adaptée à un problème de classification multi-classes.
L'optimiseur utilisé est Adam, avec un taux d'apprentissage de 0.001.
Après chaque batch, j'ai calculé la perte et la précision, et j'ai affiché la perte moyenne et la précision à la fin de chaque époque.
Finalement, j'ai lancé l'entraînement du modèle pour 5 époques et j'affiche les performances du modèle (perte et précision) après chaque époque.</p>
