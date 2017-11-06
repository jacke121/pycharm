#!/usr/bin python
# -*- coding: utf-8 -*-
# Created by Weihang Huang on 2017/8/30
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, nltk, warnings
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from wordcloud import WordCloud, STOPWORDS

PS = nltk.stem.PorterStemmer()

pd.set_option('display.max_columns', None)

df_initial = pd.read_csv("datasets/FilmRecommend/movie_metadata.csv")

df_initial.info()
print(df_initial.describe())
print(df_initial.head())

set_keywords = set()
for liste_keywords in df_initial['plot_keywords'].str.split('|').values:
    if type(liste_keywords) == float:
        continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)

print("plot_keywords", set_keywords)


def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste:
        keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords):
            continue
        for s in liste_keywords:
            if pd.notnull(s):
                keyword_count[s] += 1
    # ______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])
    keyword_occurences.sort(key=lambda x: x[1], reverse=True)
    return keyword_occurences, keyword_count


keyword_occurences, dum = count_word(df_initial, 'plot_keywords', set_keywords)
print(keyword_occurences)

doubled_entries = df_initial[df_initial.duplicated()]
df_temp = df_initial.drop_duplicates()

list_var_duplicates = ['movie_title', 'title_year', 'director_name']
liste_duplicates = df_temp['movie_title'].map(df_temp['movie_title'].value_counts() > 1)
print("Nb. of duplicate entries: {}".format(
    len(df_temp[liste_duplicates][list_var_duplicates])))
print(df_temp[liste_duplicates][list_var_duplicates].sort_values('movie_title')[31:41])

df_duplicate_cleaned = df_temp.drop_duplicates(subset=list_var_duplicates, keep='last')


# Collect the keywords
# ----------------------
def keywords_inventory(dataframe, colonne='plot_keywords'):
    PS = nltk.stem.PorterStemmer()
    keywords_roots = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        for t in s.split('|'):
            t = t.lower();
            racine = PS.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
            else:
                keywords_roots[racine] = {t}

    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    print("Nb of keywords in variable '{}': {}".format(colonne, len(category_keys)))
    return category_keys, keywords_roots, keywords_select


keywords, keywords_roots, keywords_select = keywords_inventory(df_duplicate_cleaned,
                                                               colonne='plot_keywords')

# Plot of a sample of keywords that appear in close varieties
# ------------------------------------------------------------
icount = 0
for s in keywords_roots.keys():
    if len(keywords_roots[s]) > 1:
        icount += 1
        if icount < 15:
            print(icount, keywords_roots[s], len(keywords_roots[s]))


# Replacement of the keywords by the main form
# ----------------------------------------------
def remplacement_df_keywords(df, dico_remplacement, roots=False):
    df_new = df.copy(deep=True)
    for index, row in df_new.iterrows():
        chaine = row['plot_keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'):
            clef = PS.stem(s) if roots else s
            if clef in dico_remplacement.keys():
                nouvelle_liste.append(dico_remplacement[clef])
            else:
                nouvelle_liste.append(s)
        df_new.set_value(index, 'plot_keywords', '|'.join(nouvelle_liste))
    return df_new


# Replacement of the keywords by the main keyword
# -------------------------------------------------
df_keywords_cleaned = remplacement_df_keywords(df_duplicate_cleaned, keywords_select,
                                               roots=True)

# Count of the keywords occurences
# ----------------------------------
keyword_occurences, keywords_count = count_word(df_keywords_cleaned, 'plot_keywords', keywords)
print(keyword_occurences[:5])


# get the synomyms of the word 'mot_cle'
# --------------------------------------------------------------
def get_synonymes(mot_cle):
    lemma = set()
    for ss in wordnet.synsets(mot_cle):
        for w in ss.lemma_names():
            # _______________________________
            # We just get the 'nouns':
            index = ss.name().find('.') + 1
            if ss.name()[index] == 'n': lemma.add(w.lower().replace('_', ' '))
    return lemma


# get the synomyms of the word 'mot_cle'
# --------------------------------------------------------------
def get_synonymes(mot_cle):
    lemma = set()
    for ss in wordnet.synsets(mot_cle):
        for w in ss.lemma_names():
            # _______________________________
            # We just get the 'nouns':
            index = ss.name().find('.') + 1
            if ss.name()[index] == 'n':
                lemma.add(w.lower().replace('_', ' '))
    return lemma


# Exemple of a list of synonyms given by NLTK
# ---------------------------------------------------
mot_cle = 'alien'
lemma = get_synonymes(mot_cle)
for s in lemma:
    print(' "{:<30}" in keywords list -> {} {}'.format(s, s in keywords,
                                                       keywords_count[s] if s in keywords else 0))


# check if 'mot' is a key of 'key_count' with a test on the number of occurences
# ----------------------------------------------------------------------------------
def tmp_test_keyword(mot, key_count, threshold):
    return (False, True)[key_count.get(mot, 0) >= threshold]


keyword_occurences.sort(key=lambda x: x[1], reverse=False)
key_count = dict()
for s in keyword_occurences:
    key_count[s[0]] = s[1]
# __________________________________________________________________________
# Creation of a dictionary to replace keywords by higher frequency keywords
remplacement_mot = dict()
icount = 0
for index, [mot, nb_apparitions] in enumerate(keyword_occurences):
    if nb_apparitions > 5: continue  # only the keywords that appear less than 5 times
    lemma = get_synonymes(mot)
    if len(lemma) == 0: continue  # case of the plurals
    # _________________________________________________________________
    liste_mots = [(s, key_count[s]) for s in lemma
                  if tmp_test_keyword(s, key_count, key_count[mot])]
    liste_mots.sort(key=lambda x: (x[1], x[0]), reverse=True)
    if len(liste_mots) <= 1: continue  # no replacement
    if mot == liste_mots[0][0]: continue  # replacement by himself
    icount += 1
    if icount < 8:
        print('{:<12} -> {:<12} (init: {})'.format(mot, liste_mots[0][0], liste_mots))
    remplacement_mot[mot] = liste_mots[0][0]

print(90 * '_' + '\n' + 'The replacement concerns {}% of the keywords.'
      .format(round(len(remplacement_mot) / len(keywords) * 100, 2)))

# 2 successive replacements
# ---------------------------
print('Keywords that appear both in keys and values:'.upper() + '\n' + 45 * '-')
icount = 0
for s in remplacement_mot.values():
    if s in remplacement_mot.keys():
        icount += 1
        if icount < 10: print('{:<20} -> {:<20}'.format(s, remplacement_mot[s]))

for key, value in remplacement_mot.items():
    if value in remplacement_mot.keys():
        remplacement_mot[key] = remplacement_mot[value]

# replacement of keyword varieties by the main keyword
# ----------------------------------------------------------
df_keywords_synonyms = \
    remplacement_df_keywords(df_keywords_cleaned, remplacement_mot, roots=False)
keywords, keywords_roots, keywords_select = \
    keywords_inventory(df_keywords_synonyms, colonne='plot_keywords')

# New count of keyword occurences
# -------------------------------------
new_keyword_occurences, keywords_count = count_word(df_keywords_synonyms,
                                                    'plot_keywords', keywords)
print(new_keyword_occurences[:5])


# deletion of keywords with low frequencies
# -------------------------------------------
def remplacement_df_low_frequency_keywords(df, keyword_occurences):
    df_new = df.copy(deep=True)
    key_count = dict()
    for s in keyword_occurences:
        key_count[s[0]] = s[1]
    for index, row in df_new.iterrows():
        chaine = row['plot_keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'):
            if key_count.get(s, 4) > 3: nouvelle_liste.append(s)
        df_new.set_value(index, 'plot_keywords', '|'.join(nouvelle_liste))
    return df_new


# Creation of a dataframe where keywords of low frequencies are suppressed
# -------------------------------------------------------------------------
df_keywords_occurence = \
    remplacement_df_low_frequency_keywords(df_keywords_synonyms, new_keyword_occurences)
keywords, keywords_roots, keywords_select = \
    keywords_inventory(df_keywords_occurence, colonne='plot_keywords')

# New keywords count
# -------------------
new_keyword_occurences, keywords_count = count_word(df_keywords_occurence,
                                                    'plot_keywords', keywords)
print(new_keyword_occurences[:5])

# Graph of keyword occurences
# ----------------------------
font = {'family': 'fantasy', 'weight': 'normal', 'size': 15}
mpl.rc('font', **font)

keyword_occurences.sort(key=lambda x: x[1], reverse=True)

y_axis = [i[1] for i in keyword_occurences]
x_axis = [k for k, i in enumerate(keyword_occurences)]

new_y_axis = [i[1] for i in new_keyword_occurences]
new_x_axis = [k for k, i in enumerate(new_keyword_occurences)]

df_keywords_occurence['content_rating'].unique()
dropped_var = ['aspect_ratio', 'budget', 'facenumber_in_poster',
               'content_rating', 'cast_total_facebook_likes']
df_var_cleaned = df_keywords_occurence.drop(dropped_var, axis=1)
print(df_var_cleaned.columns)

new_col_order = ['movie_title', 'title_year', 'genres', 'plot_keywords',
                 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
                 'director_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes',
                 'actor_3_facebook_likes', 'movie_facebook_likes', 'num_critic_for_reviews',
                 'num_user_for_reviews', 'num_voted_users', 'language', 'country',
                 'imdb_score', 'movie_imdb_link', 'color', 'duration', 'gross', ]
df_var_cleaned = df_var_cleaned[new_col_order]

missing_df = df_var_cleaned.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['filling_factor'] = (df_var_cleaned.shape[0]
                                - missing_df['missing_count']) / df_var_cleaned.shape[0] * 100
missing_df = missing_df.sort_values('filling_factor').reset_index(drop=True)
print(missing_df)

y_axis = missing_df['filling_factor']
x_label = missing_df['column_name']
x_axis = missing_df.index

fig = plt.figure(figsize=(11, 4))
plt.xticks(rotation=80, fontsize=14)
plt.yticks(fontsize=13)

N_thresh = 5
plt.axvline(x=N_thresh - 0.5, linewidth=2, color='r')
plt.text(N_thresh - 4.8, 30, 'filling factor \n < {}%'.format(round(y_axis[N_thresh], 1)),
         fontsize=15, family='fantasy', bbox=dict(boxstyle="round",
                                                  ec=(1.0, 0.5, 0.5),
                                                  fc=(0.8, 0.5, 0.5)))
N_thresh = 17
plt.axvline(x=N_thresh - 0.5, linewidth=2, color='g')
plt.text(N_thresh, 30, 'filling factor \n = {}%'.format(round(y_axis[N_thresh], 1)),
         fontsize=15, family='fantasy', bbox=dict(boxstyle="round",
                                                  ec=(1., 0.5, 0.5),
                                                  fc=(0.5, 0.8, 0.5)))

plt.xticks(x_axis, x_label, family='fantasy', fontsize=14)
plt.ylabel('Filling factor (%)', family='fantasy', fontsize=16)
plt.bar(x_axis, y_axis)

df_filling = df_var_cleaned.copy(deep=True)
missing_year_info = df_filling[df_filling['title_year'].isnull()][[
    'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']]
missing_year_info[:10]

# f, ax = plt.subplots(figsize=(9, 5))
# ax.plot(x_axis, y_axis, 'r-', label='before cleaning')
# ax.plot(new_x_axis, new_y_axis, 'b-', label='after cleaning')
#
# # Now add the legend with some customizations.
# legend = ax.legend(loc='upper right', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# for label in legend.get_texts():
#     label.set_fontsize('medium')
#
# plt.ylim((0, 25))
# plt.axhline(y=3.5, linewidth=2, color='k')
# plt.xlabel("keywords index", family='fantasy', fontsize=15)
# plt.ylabel("Nb. of occurences", family='fantasy', fontsize=15)
# # plt.suptitle("Nombre d'occurences des mots clés", fontsize = 18, family='fantasy')
# plt.text(3500, 4.5, 'threshold for keyword delation', fontsize=13)
# plt.show()

# f, ax = plt.subplots(figsize=(12, 9))
# #_____________________________
# # calculations of correlations
# corrmat = df_keywords_occurence.dropna(how='any').corr()
# #________________________________________
# k = 17 # number of variables for heatmap
# cols = corrmat.nlargest(k, 'num_voted_users')['num_voted_users'].index
# cm = np.corrcoef(df_keywords_occurence[cols].dropna(how='any').values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
#                  fmt='.2f', annot_kws={'size': 10}, linewidth = 0.1, cmap = 'coolwarm',
#                  yticklabels=cols.values, xticklabels=cols.values)
# f.text(0.5, 0.93, "Correlation coefficients", ha='center', fontsize = 18, family='fantasy')
# plt.show()
# missing_df = df_initial.isnull().sum(axis=0).reset_index()
# missing_df.columns = ['column_name', 'missing_count']
# missing_df['filling_factor'] = (df_initial.shape[0]
#                                 - missing_df['missing_count']) / df_initial.shape[0] * 100
# print(missing_df.sort_values('filling_factor').reset_index(drop = True))

# # _____________________________________________
# # Function that control the color of the words
# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # WARNING: the scope of variables is used to get the value of the "tone" variable
# # I could not find the way to pass it as a parameter of "random_color_func()"
# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# def random_color_func(word=None, font_size=None, position=None,
#                       orientation=None, font_path=None, random_state=None):
#     h = int(360.0 * tone / 255.0)
#     s = int(100.0 * 255.0 / 255.0)
#     l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
#     return "hsl({}, {}%, {}%)".format(h, s, l)
#
#
# # _____________________________________________
# # UPPER PANEL: WORDCLOUD
# fig = plt.figure(1, figsize=(18, 13))
# ax1 = fig.add_subplot(2, 1, 1)
# # _______________________________________________________
# # I define the dictionary used to produce the wordcloud
# words = dict()
# trunc_occurences = keyword_occurences[0:50]
# for s in trunc_occurences:
#     words[s[0]] = s[1]
# tone = 55.0  # define the color of the words
# # ________________________________________________________
# wordcloud = WordCloud(width=1000, height=300, background_color='black',
#                       max_words=1628, relative_scaling=1,
#                       color_func=random_color_func,
#                       normalize_plurals=False)
# wordcloud.generate_from_frequencies(words)
# ax1.imshow(wordcloud, interpolation="bilinear")
# ax1.axis('off')
# # _____________________________________________
# # LOWER PANEL: HISTOGRAMS
# ax2 = fig.add_subplot(2, 1, 2)
# y_axis = [i[1] for i in trunc_occurences]
# x_axis = [k for k, i in enumerate(trunc_occurences)]
# x_label = [i[0] for i in trunc_occurences]
# plt.xticks(rotation=85, fontsize=15)
# plt.yticks(fontsize=15)
# plt.xticks(x_axis, x_label)
# plt.ylabel("Nb. of occurences", fontsize=18, labelpad=10)
# ax2.bar(x_axis, y_axis, align='center', color='g')
# # _______________________
# plt.title("Keywords popularity", bbox={'facecolor': 'k', 'pad': 5}, color='w', fontsize=25)
# plt.show()
