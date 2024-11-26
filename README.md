<h1>atelier 2 Deep learning</h1>
<H3>1. Establish a CNN Architecture (Based on Pytorch Library) to classify MINST Dataset, by
defining layers (Convolution, pooling, fully connect layer), the hyper-parameters (Kernels,
Padding , stride, optimizers, regularization, etc) and running the model in GPU mode.</H3>
<p>J'ai chargé les données MNIST en créant deux fonctions spécifiques :

load_mnist_images(filename) : Cette fonction charge les images à partir du fichier binaire, les normalise entre 0 et 1, puis les retourne sous forme de tableau numpy.
load_mnist_labels(filename) : Cette fonction charge les labels associés aux images (les chiffres de 0 à 9) à partir du fichier binaire.
J'ai ensuite défini les chemins vers les fichiers MNIST et utilisé ces fonctions pour charger les images et les labels des ensembles d'entraînement et de test.

J'ai préparé les données pour PyTorch en convertissant les images et les labels en tensors PyTorch. Pour les images, j'ai ajouté une dimension supplémentaire pour le canal (gris, donc une seule dimension).
Ensuite, j'ai créé des Datasets PyTorch pour l'entraînement et les tests à partir des images et des labels. J'ai également créé des DataLoaders pour charger les données en batchs pendant l'entraînement et les tests.

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
<h3>2. Do the same thing with Faster R-CNN.</h3>
<p>Chargement des images (load_mnist_images) :
J'ai ouvert le fichier des images et extrait les informations essentielles comme le nombre d'images, la taille (en pixels) de chaque image, puis j'ai chargé les données binaires.
J'ai ensuite redimensionné les données sous forme de tableau numpy, où chaque image a une forme de (28, 28), et j'ai normalisé les valeurs entre 0 et 1 en les divisant par 255.
Chargement des labels (load_mnist_labels) :
J'ai ouvert le fichier des labels et extrait le nombre de labels, puis j'ai chargé les données associées aux labels sous forme de tableau numpy, où chaque label correspond à un chiffre entre 0 et 9.
J'ai ensuite défini les chemins vers les fichiers des images et des labels MNIST, et j'ai utilisé les fonctions pour charger les images et les labels des ensembles d'entraînement et de test.

Préparation des données pour PyTorch
Une fois les données chargées, j'ai effectué les étapes suivantes pour les préparer à l'utilisation dans PyTorch :

Conversion en tensors PyTorch :
J'ai converti les images et les labels en tensors PyTorch afin qu'ils puissent être utilisés dans le modèle.
Pour les images, j'ai ajouté une dimension supplémentaire pour représenter le canal des couleurs (en l'occurrence, une seule dimension puisque les images sont en niveaux de gris).
Création de datasets et DataLoaders :
J'ai créé des Datasets PyTorch en utilisant TensorDataset, en combinant les images et les labels.
Ensuite, j'ai utilisé ces datasets pour créer des DataLoaders, qui me permettent de charger les données en batchs pendant l'entraînement et les tests.
Définition du modèle CNN
J'ai ensuite créé un modèle de réseau de neurones convolutionnel (CNN) pour classer les images de chiffres. Le modèle est défini dans une classe CNNModel qui hérite de nn.Module. Voici la structure du modèle :

Couches Convolutionnelles :

conv1 : J'ai défini une première couche convolutionnelle avec 32 filtres de taille 3x3, un stride de 1 et un padding de 1 pour conserver la taille des images à (28, 28).
conv2 : J'ai ajouté une deuxième couche convolutionnelle avec 64 filtres, qui réduit la taille des images à (14, 14) après application de la couche de pooling.
conv3 : J'ai ajouté une troisième couche convolutionnelle avec 128 filtres, qui réduit encore la taille des images à (7, 7).
Couches de Pooling :

Après chaque couche convolutionnelle, j'ai utilisé une couche de max pooling avec un noyau de taille 2x2 pour réduire la taille des images de moitié.
Couches Fully Connected :

Après les convolutions et les opérations de pooling, j'ai aplati les données (en utilisant view), ce qui transforme les images de forme (batch_size, 128, 7, 7) en un vecteur de taille (batch_size, 128 * 7 * 7).
J'ai défini une première couche fully connected fc1 avec 512 neurones et une seconde couche fc2 avec 10 neurones, correspondant aux 10 classes du dataset MNIST.
Fonction de passage avant (forward)
La fonction forward définit la façon dont les données traversent le modèle pendant l'entraînement ou la prédiction. Voici ce qui se passe dans cette fonction :

Les données passent successivement dans les couches convolutionnelles et de pooling.
Après les convolutions, les données sont aplaties pour être passées dans les couches fully connected.
Les données sont ensuite transmises à travers les couches fully connected pour produire une sortie avec 10 neurones, correspondant aux probabilités de chaque classe (chiffre entre 0 et 9).
Entraînement du modèle
J'ai ensuite défini une fonction d'entraînement train_model pour entraîner le modèle avec les données MNIST. Voici comment cela fonctionne :

Déplacement du modèle sur le GPU :

Si un GPU est disponible, j'ai déplacé le modèle sur le GPU pour accélérer l'entraînement.
Boucle d'entraînement :

J'ai itéré sur les époques d'entraînement. À chaque époque, j'ai parcouru les batches d'images et de labels dans le DataLoader.
Pour chaque batch, j'ai effectué les étapes suivantes :
J'ai déplacé les images et les labels sur le GPU.
J'ai passé les images à travers le modèle pour obtenir les prédictions.
J'ai calculé la perte en utilisant la fonction CrossEntropyLoss, qui est adaptée à la classification multi-classes.
J'ai effectué la rétropropagation en appelant loss.backward() pour calculer les gradients.
J'ai mis à jour les poids du modèle en appelant optimizer.step().
Affichage des statistiques :

J'ai calculé la précision en comparant les prédictions avec les labels réels.
J'ai affiché la perte et la précision après chaque époque.
</p>
<h3>3. Compare the two models (By using several metrics (Accuracy, F1 score, Loss, Training time))</h3>
Le modèle 2 (RCNNClassifier) a obtenu les meilleurs résultats avec une précision de 99,06%, un score F1 de 0,9906 et une perte de 0,0289, surpassant le modèle 1 (CNNModel), qui a atteint une précision de 98,90%, un score F1 de 0,9890 et une perte de 0,0290. Cependant, le modèle 2 a nécessité un temps d'exécution nettement plus long (2238,45 secondes contre 587,17 secondes pour le modèle 1).


