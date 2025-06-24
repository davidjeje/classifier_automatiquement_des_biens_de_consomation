import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # Dictionnaire multilingue
# Assure-toi d'avoir ces téléchargements
nltk.download('punkt')
nltk.download('stopwords')
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import re
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModel
import torch
import tensorflow_hub as hub
import tensorflow as tf
# Chargement modèle + tokenizer (exemple : bert-base-uncased)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
from PIL import Image
import os 
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageOps

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)                  # Enlever les chiffres
    text = re.sub(r'[^\w\s]', '', text)              # Enlever la ponctuation
    text = re.sub(r'\s+', ' ', text).strip()         # Enlever les espaces multiples
    return text

def generate_wordclouds_by_category_with_extraction(df, text_column='description', category_column='product_category_tree'):
    """
    Génère et affiche un Word Cloud pour chaque catégorie principale extraite 
    depuis une colonne de type hiérarchique ('product_category_tree').

    Paramètres :
    - df : pandas.DataFrame contenant les colonnes de texte et de catégorie hiérarchique
    - text_column : nom de la colonne contenant le texte (ex: 'description')
    - category_column : nom de la colonne contenant les chaînes de catégories hiérarchiques (ex: 'product_category_tree')
    """
    def extract_category(text):
        if isinstance(text, str):
            match = re.search(r'"([^"]+?)\s*>>', text)
            if match:
                return match.group(1).strip()
        return None

    # Ajout d'une colonne temporaire avec la catégorie extraite
    df = df.copy()
    df['extracted_category'] = df[category_column].apply(extract_category)

    categories = df['extracted_category'].dropna().unique()

    for cat in categories:
        subset = df[df['extracted_category'] == cat][text_column].dropna()
        if subset.empty:
            continue
        combined_text = subset.apply(clean_text).str.cat(sep=' ')
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud - {cat}")
        plt.show()

def extract_unique_categories(df, column='product_category_tree'):
    """
    Extrait le premier texte entre guillemets et avant '>>' pour chaque ligne de la colonne spécifiée,
    et retourne un DataFrame contenant ces valeurs uniques.
    """
    extracted = []

    for text in df[column].dropna():
        # Recherche d'un texte entre guillemets, suivi de '>>'
        match = re.search(r'"([^"]+?)\s*>>', text)
        if match:
            extracted.append(match.group(1).strip())

    # Supprimer les doublons tout en conservant l’ordre d’apparition
    unique_values = list(dict.fromkeys(extracted))

    # Retourner un DataFrame avec une seule colonne
    return pd.DataFrame(unique_values, columns=['extracted_category'])

def extract_category(text):
    if pd.isna(text):
        return None
    match = re.search(r'"(.*?)\s*>>', text)
    if match:
        return match.group(1).strip()
    return None

def detect_language_safe(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

# Création du tokenizer avec l'expression régulière
regexp_tokenizer = RegexpTokenizer(r'\w+')

def tokenize_english_descriptions(df, desc_col='description'):
    df['lang'] = df[desc_col].astype(str).apply(detect_language_safe)
    df['tokens'] = df.apply(
        lambda row: regexp_tokenizer.tokenize(row[desc_col].lower()) if row['lang'] == 'en' else [],
        axis=1
    )
    return df

def afficher_tokens(df, index, col='description'):
    print("Langue :", df.loc[index, 'lang'])
    
    if col == 'description':
        print("\nDescription originale :\n", df.loc[index, col])
    elif col == 'product_name':
        print("\nTitre original :\n", df.loc[index, col])
    else:
        print(f"\nContenu de la colonne '{col}' :\n", df.loc[index, col])
    
    print("\nTokens :\n", df.loc[index, 'tokens'])


def get_word_frequencies(df, tokens_col='tokens'):
    # Aplatir toutes les listes de tokens en une seule grande liste
    all_tokens = [token for tokens in df[tokens_col] for token in tokens]
    
    # Compter la fréquence de chaque mot
    word_counts = Counter(all_tokens)
    
    # Convertir en DataFrame trié
    freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])
    freq_df = freq_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
    
    return freq_df

def plot_word_frequencies(freq_df, top_n=20, title='Top Word Frequencies'):
    """
    Affiche un graphique à barres des mots les plus fréquents.

    Parameters:
    - freq_df : DataFrame retourné par get_word_frequencies
    - top_n : nombre de mots à afficher
    - title : titre du graphique
    """
    # Sélection des top_n mots les plus fréquents
    top_words = freq_df.head(top_n)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='word',
        y='frequency',
        hue='word',  # nécessaire pour autoriser l'utilisation de 'palette'
        data=top_words,
        palette='viridis',
        legend=False
    )

    plt.title(title, fontsize=16)
    plt.xlabel('Mot', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def get_word_frequencies_filtered(df, token_col='tokens', language='english'):
    """
    Compte la fréquence des mots dans une colonne de tokens en filtrant les stopwords.

    Parameters:
    - df : DataFrame contenant une colonne de tokens
    - token_col : nom de la colonne contenant des listes de mots
    - language : langue pour les stopwords (par défaut 'english')

    Returns:
    - DataFrame avec les mots non-stopwords et leur fréquence triée décroissante
    """
    # Vérifier que les stopwords sont bien téléchargés
    nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words(language))
    
    # Récupération de tous les mots sauf stopwords
    words = [
        word for tokens in df[token_col] for word in tokens
        if word.lower() not in stop_words
    ]
    
    # Comptage
    word_counts = Counter(words)
    
    # Conversion en DataFrame
    freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])
    freq_df = freq_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
    
    return freq_df

def get_word_frequencies_lemmatized(df, token_col='tokens', language='english'):
    """
    Compte la fréquence des lemmes non stopwords dans une colonne de tokens.
    """
    # Téléchargement nécessaire
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    stop_words = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatisation et filtrage
    words = [
        lemmatizer.lemmatize(word.lower())
        for tokens in df[token_col] for word in tokens
        if word.lower() not in stop_words
    ]

    # Comptage des fréquences
    word_counts = Counter(words)
    
    # Conversion en DataFrame
    freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])
    freq_df = freq_df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
    
    return freq_df

def get_lemmatized_frequencies_by_description(df, desc_col='description'):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Liste pour stocker les compteurs par description
    all_frequencies = []

    for description in df[desc_col].astype(str):
        # Tokenization
        tokens = word_tokenize(description.lower())
        # Filtrage stopwords et caractères non alphanumériques
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        # Lemmatisation
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        # Comptage
        freq = Counter(lemmatized)
        all_frequencies.append(freq)

    # Création d’un DataFrame : une ligne par description, une colonne par mot
    df_bow = pd.DataFrame(all_frequencies).fillna(0).astype(int)
    
    return df_bow

def plot_bow_heatmap(df_bow, top_n_words=20, nb_resultats=10, type_contenu='descriptions'):
    """
    Affiche une heatmap des top_n_words les plus fréquents dans les nb_resultats premières lignes
    (soit des descriptions soit des titres).

    Paramètres :
    - df_bow : DataFrame issu du bag of words (une ligne = une description ou un titre, une colonne = un mot)
    - top_n_words : nombre de mots les plus fréquents à afficher (colonnes)
    - nb_resultats : nombre de lignes à afficher (lignes = titres ou descriptions)
    - type_contenu : chaîne indiquant ce qui est affiché, 'descriptions' ou 'titres'
    """
    # Exclure les colonnes non lexicales
    df_lexical = df_bow.drop(columns=['key', 'feature'], errors='ignore')
    
    # Sélection des mots les plus fréquents
    top_words = df_lexical.sum().sort_values(ascending=False).head(top_n_words).index.tolist()
    
    # Sous-échantillonnage des lignes
    subset = df_lexical[top_words].head(nb_resultats)
    
    # Création de la heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(subset, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
    plt.title(f"Heatmap : fréquence des {top_n_words} mots les plus fréquents dans les {nb_resultats} premiers {type_contenu}")
    plt.xlabel("Mots")
    plt.ylabel(type_contenu.capitalize())
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def apply_tfidf(df_bow, id_columns=['key', 'feature']):
    """
    Applique la pondération TF-IDF sur un DataFrame de fréquences par document.
    
    Paramètres :
        df_bow : DataFrame avec colonnes d'identifiants + mots
        id_columns : Liste des colonnes à ne pas transformer
        
    Retourne :
        df_tfidf : DataFrame avec les mêmes index + valeurs TF-IDF
    """
    # Identifier les colonnes à supprimer si elles existent
    id_cols_presentes = [col for col in id_columns if col in df_bow.columns]
    
    # Séparer les colonnes identifiants des colonnes de mots
    X_counts = df_bow.drop(columns=id_cols_presentes)
    
    # Appliquer TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    
    # Reconstituer le DataFrame avec les mêmes index
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=X_counts.columns, index=df_bow.index)
    
    # Réintégrer les colonnes identifiants présentes
    for col in id_cols_presentes:
        df_tfidf[col] = df_bow[col].values
    
    return df_tfidf

def plot_top_tfidf_words(tfidf_matrix, feature_names, doc_index=0, top_n=10, source="description"):
    """
    Affiche les top mots TF-IDF d'un document donné, selon qu'il s'agit d'une description ou d'un titre.

    Parameters:
    - tfidf_matrix: matrice TF-IDF (numpy array ou sparse matrix)
    - feature_names: liste des mots (colonnes de la matrice TF-IDF)
    - doc_index: index du document à visualiser
    - top_n: nombre de mots à afficher
    - source: "description" ou "titre" (utilisé dans le titre du graphique)
    """
    row = tfidf_matrix[doc_index].flatten()
    top_indices = row.argsort()[-top_n:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = row[top_indices]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_scores, y=top_words, hue=top_words, dodge=False, palette='viridis', legend=False)
    plt.xlabel("Score TF-IDF")
    plt.title(f"Top {top_n} mots TF-IDF pour {source} {doc_index}")
    plt.tight_layout()
    plt.show()

def reduce_tsne(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    return tsne.fit_transform(X.toarray())

def reduce_tsne2(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    return tsne.fit_transform(X)

def reduce_tsne_bert_use(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    return tsne.fit_transform(X)

def cluster_kmeans(X_2d, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X_2d)

def plot_tsne(X_2d, labels, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette='tab10', legend='full')
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def vectorize_text(tokens, model):
    valid_tokens = [w for w in tokens if w in model.wv]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in valid_tokens], axis=0)

# Chargement modèle + tokenizer (exemple : bert-base-uncased)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

def bert_vectorize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Moyenne des embeddings du dernier hidden state (token embeddings)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Chargement modèle USE (en local ou via URL)
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def use_vectorize(text):
    embeddings = use_model([text])
    return embeddings[0].numpy()

def plot_2D_clusters(X_2D, labels, title):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2D[:,0], X_2D[:,1], c=labels, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.show()

def get_image_size(path):
    try:
        with Image.open(os.path.join(image_dir, filename)) as img:
            return img.size  # (largeur, hauteur)
    except Exception as e:
        print(f"Erreur avec {filename} : {e}")
        return None


# 📁 Paramètres
IMAGE_DIR = "data/images"  # Dossier contenant les images
IMAGE_SIZE = (224, 224)  # Pour CNN
N_CLUSTERS = 7  # Nombre de catégories réelles
RANDOM_STATE = 42

# 🔧 Prétraitement générique pour CNN
def preprocess_image_cnn(path):
    img = load_img(path, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# 📷 SIFT Feature Extraction
def extract_features_sift(path, max_features=128):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create(nfeatures=max_features)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            return np.zeros((max_features, 128)).flatten()
        descriptors = descriptors[:max_features]
        if descriptors.shape[0] < max_features:
            padding = np.zeros((max_features - descriptors.shape[0], descriptors.shape[1]))
            descriptors = np.vstack([descriptors, padding])
        return descriptors.flatten()
    except:
        return np.zeros((max_features, 128)).flatten()

# 📷 SURF Feature Extraction
def extract_features_surf(path, max_features=128):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        keypoints, descriptors = surf.detectAndCompute(img, None)
        if descriptors is None:
            return np.zeros((max_features, 64)).flatten()
        descriptors = descriptors[:max_features]
        if descriptors.shape[0] < max_features:
            padding = np.zeros((max_features - descriptors.shape[0], descriptors.shape[1]))
            descriptors = np.vstack([descriptors, padding])
        return descriptors.flatten()
    except:
        return np.zeros((max_features, 64)).flatten()


# 📷 Prétraitement et extraction SIFT/ORB
def extract_features_orb(path, max_features=128):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is None:
            return np.zeros((max_features, 32)).flatten()
        descriptors = descriptors[:max_features]
        if descriptors.shape[0] < max_features:
            padding = np.zeros((max_features - descriptors.shape[0], descriptors.shape[1]))
            descriptors = np.vstack([descriptors, padding])
        return descriptors.flatten()
    except:
        return np.zeros((max_features, 32)).flatten()

# 🧠 CNN Feature extraction (ResNet50 sans couche finale)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
# base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
def extract_features_cnn(path):
    img = preprocess_image_cnn(path)
    img = np.expand_dims(img, axis=0)
    features = base_model.predict(img, verbose=0)
    return features.flatten()

# 🎨 Réduction de dimension & Clustering & Visualisation
def run_analysis(feature_matrix, labels, method_name):
    print(f"\n🔍 Méthode : {method_name}")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    reduced = tsne.fit_transform(feature_matrix)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    predicted_clusters = kmeans.fit_predict(feature_matrix)

    ari = adjusted_rand_score(labels, predicted_clusters)
    print(f"🎯 ARI ({method_name}): {ari:.4f}")

    # 🔵 Graphique 1 : couleurs = clusters KMeans
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=predicted_clusters, palette='tab10')
    plt.title(f'{method_name} - Clusters KMeans')

    # 🟢 Graphique 2 : couleurs = catégories réelles
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='Set2')
    plt.title(f'{method_name} - Catégories Réelles')
    plt.tight_layout()
    plt.show()

    return ari

from sklearn.preprocessing import LabelEncoder

def prepare_labels(df):
    # 🏷️ Extraire les vraies catégories à partir de la colonne product_category_tree
    df["extracted_category"] = df["product_category_tree"].apply(extract_category)
    
    # ✅ Vérifications de cohérence
    assert len(df["image_path"]) == len(df["extracted_category"]), \
        "❌ Mismatch: image_path and extracted_category must have the same length"
    assert not df["extracted_category"].isna().any(), \
        "❌ Missing values found in extracted_category"

    # 🔢 Encodage des étiquettes en entiers
    label_encoder = LabelEncoder()
    label_encoder.fit(df["extracted_category"])
    encoded_labels = label_encoder.transform(df["extracted_category"])
    
    return encoded_labels, label_encoder

def run_all_methods(df):
    # 🎯 Préparation des labels
    true_labels, label_encoder = prepare_labels(df)
    
    results = []

    # SIFT
    print("\n⏳ Extraction SIFT...")
    sift_features = np.array([extract_features_sift(p) for p in tqdm(df['image_path'])])
    ari_sift = run_analysis(sift_features, true_labels, "SIFT")
    results.append(("SIFT", ari_sift))

    # 📋 Résumé des résultats
    results_df = pd.DataFrame(results, columns=["Méthode", "ARI"])
    results_df.sort_values(by="ARI", ascending=False, inplace=True)
    print("\n📈 Résultats comparatifs :")
    print(results_df)
    return results_df  # <== ici aussi

def run_all_methods2(df):
    # 🎯 Préparation des labels
    true_labels, label_encoder = prepare_labels(df)
    
    results = []

    # ORB
    print("\n⏳ Extraction ORB...")
    orb_features = np.array([extract_features_orb(p) for p in tqdm(df['image_path'])])
    ari_orb = run_analysis(orb_features, true_labels, "ORB")
    results.append(("ORB", ari_orb))

    # CNN
    print("\n⏳ Extraction CNN (ResNet50)...")
    cnn_features = np.array([extract_features_cnn(p) for p in tqdm(df['image_path'])])
    ari_cnn = run_analysis(cnn_features, true_labels, "CNN (ResNet50)")
    results.append(("CNN (ResNet50)", ari_cnn))

    # 📋 Résumé des résultats
    results_df = pd.DataFrame(results, columns=["Méthode", "ARI"])
    results_df.sort_values(by="ARI", ascending=False, inplace=True)
    print("\n📈 Résultats comparatifs :")
    print(results_df)

    return results_df  # <== ici aussi

def run_all_methods3(df):
    # 🎯 Préparation des labels
    true_labels, label_encoder = prepare_labels(df)
    
    results = []

    # SURF
    print("\n⏳ Extraction SURF...")
    surf_features = np.array([extract_features_surf(p) for p in tqdm(df['image_path'])])
    ari_surf = run_analysis(surf_features, true_labels, "SURF")
    results.append(("SURF", ari_surf))

    # 📋 Résumé des résultats
    results_df = pd.DataFrame(results, columns=["Méthode", "ARI"])
    results_df.sort_values(by="ARI", ascending=False, inplace=True)
    print("\n📈 Résultats comparatifs :")
    print(results_df)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Modèles CNN de base
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2


def prepare_dataframe(df, image_dir):
    """Ajoute les chemins complets aux images et encode les labels (sans les transformer en entiers)."""
    le = LabelEncoder()
    df["label"] = df["extracted_category"]  # On garde les labels en str
    df["filename"] = df["image"].apply(lambda x: os.path.join(image_dir, x))
    return df, le.fit(df["label"])


def create_data_generators(df, input_shape, batch_size):
    """Crée les générateurs de données avec augmentation pour l'entraînement."""
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col="filename",
        y_col="label",
        target_size=input_shape[:2],
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True
    )

    test_gen = test_datagen.flow_from_dataframe(
        df_test,
        x_col="filename",
        y_col="label",
        target_size=input_shape[:2],
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, test_gen, df_train, df_test


def build_model(base_model_class, input_shape, num_classes):
    """Construit un modèle CNN basé sur un backbone pré-entraîné (VGG, ResNet, etc.)."""
    base_model = base_model_class(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def evaluate_model(model, generator, le, title, cmap):
    """Évalue le modèle et affiche une matrice de confusion + rapport de classification."""
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Matrice de confusion - {title}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()

    print(f"\n📊 Rapport de classification - {title}")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

def evaluate_model_2(model, data_gen, label_encoder, title="Evaluation", cmap="Blues"):
    """Évalue un modèle CNN et affiche la matrice de confusion 7x7."""
    # Récupération des vraies classes et prédictions
    y_true = data_gen.classes
    y_pred_probs = model.predict(data_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Noms des classes
    class_names = label_encoder.classes_
    
    # 📊 Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion - {title}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()

    # 🧾 Rapport de classification
    print(f"\n🧾 Rapport de classification - {title} :\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

def train_and_evaluate_model(base_model_class, model_name, train_gen, test_gen, input_shape, num_classes, le, epochs=10):
    """Entraîne un modèle CNN et évalue ses performances sur les sets d'entraînement et test."""
    print(f"\n🔧 Entraînement du modèle {model_name}...")
    model = build_model(base_model_class, input_shape, num_classes)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model.fit(train_gen, validation_data=test_gen, epochs=epochs, callbacks=[early_stop])

    evaluate_model(model, train_gen, le, f"{model_name} - TRAIN", "Blues")
    evaluate_model(model, test_gen, le, f"{model_name} - TEST", "Greens")

    return model


def reduce_pca(X, n_components=50):
    """Réduit la dimension via PCA avant t-SNE."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)

def find_best_k(X, max_k=15):
    """Affiche la méthode du coude pour choisir k."""
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title("Méthode du coude")
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Inertie (Within-cluster sum of squares)")
    plt.grid(True)
    plt.show()

# def full_clustering_pipeline(X, y_true, n_clusters=7, name="experiment", logs=[]):
#     """Pipeline complet : PCA → t-SNE → KMeans → ARI → CSV"""
#     from sklearn.manifold import TSNE
#     import time, csv
#     from datetime import datetime

#     start = time.time()

#     # PCA (réduction initiale pour t-SNE)
#     X_pca = reduce_pca(X, n_components=50)

#     # t-SNE
#     tsne = TSNE(n_components=2, random_state=0)
#     X_tsne = tsne.fit_transform(X_pca)

#     # KMeans
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#     labels = kmeans.fit_predict(X)

#     # ARI
#     ari = adjusted_rand_score(y_true, labels)
#     elapsed = time.time() - start

#     print(f"{name} → ARI = {ari:.4f}, Temps = {elapsed:.2f}s")

#     # Sauvegarde CSV
#     now = datetime.now().strftime("%Y-%m-%d %H:%M")
#     logs.append({
#         "nom_tentative": name,
#         "date_heure": now,
#         "ARI": ari,
#         "temps_sec": elapsed
#     })

#     with open("resultats_kmeans.csv", mode="a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=logs[0].keys())
#         if f.tell() == 0:
#             writer.writeheader()
#         writer.writerows(logs)

#     return ari

def full_clustering_pipeline(X, y_true, n_clusters=7, name="experiment", logs=[], show_plot=True):
    """
    Pipeline complet : PCA → t-SNE → KMeans → ARI → CSV → Affichage
    """
    import time, csv
    from datetime import datetime
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    start = time.time()

    # Étape 1 : PCA
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)

    # Étape 2 : t-SNE sur la sortie PCA
    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    # Étape 3 : Clustering KMeans sur les vecteurs d’origine (non réduits)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    # Étape 4 : Score ARI
    ari = adjusted_rand_score(y_true, labels)
    elapsed = time.time() - start

    print(f"{name} → ARI = {ari:.4f}, Temps = {elapsed:.2f}s")

    # Étape 5 : Affichage t-SNE
    if show_plot:
        plt.figure(figsize=(10, 5))

        # Prédictions
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab10', s=50)
        plt.title(f"KMeans clusters ({n_clusters}) - {name}")
        plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2")

        # Vraies classes
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_true, palette='tab20', s=50)
        plt.title("Vraies catégories")
        plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2")

        plt.tight_layout()
        plt.show()

    # Étape 6 : Sauvegarde CSV
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    logs.append({
        "nom_tentative": name,
        "date_heure": now,
        "ARI": ari,
        "temps_sec": elapsed
    })

    with open("resultats_kmeans.csv", mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(logs)

    return ari


# Fonction pour filtrer les stopwords sur une chaîne de texte
def filter_stopwords_from_text(text, lang='english'):
    if not isinstance(text, str):
        return []
    
    # Tokenisation en mots
    tokens = word_tokenize(text.lower())  # en minuscules
    
    # Récupération des stopwords et ponctuations
    stop_words = set(stopwords.words(lang))
    punctuation = set(string.punctuation)
    
    # Filtrage des tokens : pas de stopwords, pas de ponctuation, mots non vides
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation and token.strip() != '']
    
    return filtered_tokens

def add_filtered_tokens_column(df, source_col='description', new_col='tokens_filtered', lang='english'):
    df[new_col] = df[source_col].apply(lambda x: filter_stopwords_from_text(x, lang=lang))
    return df

def afficher_tokens_supprimes(df, index, col_original='tokens_description', col_filtrée='tokens_filtered'):
    tokens_originaux = df.loc[index, col_original]
    tokens_filtres = df.loc[index, col_filtrée]

    tokens_supprimes = [token for token in tokens_originaux if token not in tokens_filtres]
    
    print(f"Tokens originaux (ligne {index}):\n{tokens_originaux}\n")
    print(f"Tokens filtrés (ligne {index}):\n{tokens_filtres}\n")
    print(f"❌ Tokens supprimés (ligne {index}):\n{tokens_supprimes}")

def stem_tokens_column(df, source_col, new_col):
    stemmer = PorterStemmer()
    df[new_col] = df[source_col].apply(
        lambda tokens: [stemmer.stem(token) for token in tokens if isinstance(token, str)]
    )
    return df

def afficher_exemple_stem(df, ligne):
    print(f"📄 Description originale (ligne {ligne}):\n{df.loc[ligne, 'description']}\n")
    print(f"📝 Titre original (ligne {ligne}):\n{df.loc[ligne, 'product_name']}\n")

    print(f"🧪 Tokens filtrés description:\n{df.loc[ligne, 'tokens_description_filtered']}\n")
    print(f"🌱 Tokens stemmés description:\n{df.loc[ligne, 'tokens_description_stemmed']}\n")

    print(f"🧪 Tokens filtrés titre:\n{df.loc[ligne, 'tokens_titre_filtered']}\n")
    print(f"🌱 Tokens stemmés titre:\n{df.loc[ligne, 'tokens_titre_stemmed']}")

def is_low_contrast(img_rgb, threshold=100):
    """Renvoie True si au moins un canal a une plage < seuil."""
    for i in range(3):  # R, G, B
        c_min = img_rgb[:, :, i].min()
        c_max = img_rgb[:, :, i].max()
        if (c_max - c_min) < threshold:
            return True
    return False 

def plot_image_and_histogram(image, ax_img, ax_hist, title):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax_img.imshow(img_rgb)
    ax_img.set_title(title)
    ax_img.axis('off')

    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0,256])
        ax_hist.plot(hist, color=col)
    ax_hist.set_xlim([0,256])
    ax_hist.set_title('Histogramme')

def apply_contrast_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final 

def equalize_hist_color(image):
    # Convertir en LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Égalisation de l'histogramme sur le canal L
    l_eq = cv2.equalizeHist(l)

    lab_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return image_eq

# Égalisation d’histogramme avec PIL.ImageOps.equalize
def equalize_with_pil(image_cv2):
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    pil_eq = ImageOps.equalize(pil_img)
    img_eq = cv2.cvtColor(np.array(pil_eq), cv2.COLOR_RGB2BGR)
    return img_eq

# Égalisation d’histogramme avec OpenCV (canal L en LAB)
def equalize_with_cv2(image_cv2):
    lab = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return image_eq

def rotate_image(image, angle_deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Appliquer rotation
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Fonction pour afficher image + histogramme
def plot_gray_image_and_histogram(image, ax_img, ax_hist, title):
    ax_img.imshow(image, cmap='gray')
    ax_img.set_title(title)
    ax_img.axis('off')

    # Histogramme
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax_hist.plot(hist, color='black')
    ax_hist.set_xlim([0, 256])
    ax_hist.set_title("Histogramme")
