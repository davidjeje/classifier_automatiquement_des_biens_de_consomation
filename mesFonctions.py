import pandas as pd
import numpy as np
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

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

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
