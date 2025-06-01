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
# from transformers import AutoTokenizer, AutoModel
import torch
# import tensorflow_hub as hub

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

def afficher_tokens(df, index):
    print("Langue :", df.loc[index, 'lang'])
    print("\nDescription originale :\n", df.loc[index, 'description'])
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

def plot_bow_heatmap(df_bow, top_n_words=20, nb_descriptions=10):
    """
    Affiche une heatmap des top_n_words les plus fréquents dans les nb_descriptions premières descriptions.
    
    Paramètres :
    - df_bow : DataFrame issu du bag of words (une ligne = une description, une colonne = un mot)
    - top_n_words : nombre de mots les plus fréquents à afficher
    - nb_descriptions : nombre de descriptions à afficher en lignes
    """
    # Exclure les colonnes non lexicales
    df_lexical = df_bow.drop(columns=['key', 'feature'], errors='ignore')
    
    # Sélection des mots les plus fréquents
    top_words = df_lexical.sum().sort_values(ascending=False).head(top_n_words).index.tolist()
    
    # Sous-échantillonnage des descriptions
    subset = df_lexical[top_words].head(nb_descriptions)
    
    # Création de la heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(subset, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
    plt.title(f"Heatmap : fréquence des {top_n_words} mots les plus fréquents dans les {nb_descriptions} premières descriptions")
    plt.xlabel("Mots")
    plt.ylabel("Descriptions")
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
    # Séparer les colonnes identifiants des colonnes mots
    X_counts = df_bow.drop(columns=id_columns)
    
    # Appliquer TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    
    # Reconstituer le DataFrame avec les mêmes colonnes
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=X_counts.columns)
    
    # Réintégrer les colonnes identifiants
    for col in id_columns:
        df_tfidf[col] = df_bow[col].values
    
    return df_tfidf

def plot_top_tfidf_words(tfidf_matrix, feature_names, doc_index=0, top_n=10):
    row = tfidf_matrix[doc_index].flatten()
    top_indices = row.argsort()[-top_n:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = row[top_indices]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_scores, y=top_words, hue=top_words, dodge=False, palette='viridis', legend=False)
    plt.xlabel("Score TF-IDF")
    plt.title(f"Top {top_n} mots TF-IDF pour la description {doc_index}")
    plt.tight_layout()
    plt.show()

def reduce_tsne(X):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    return tsne.fit_transform(X.toarray())

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
# use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def use_vectorize(text):
    embeddings = use_model([text])
    return embeddings[0].numpy()

def plot_2D_clusters(X_2D, labels, title):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2D[:,0], X_2D[:,1], c=labels, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.show()
