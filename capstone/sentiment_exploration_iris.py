#%%
import numpy as np
import pandas as pd
import pickle
import string
import warnings
warnings.filterwarnings("ignore")

#%%
data = pd.read_csv('/Users/Hengyu/Desktop/Beer_Data/beer-ratings/train.csv')
# 10 NaN value in the dataframe, therefore removed
data = data[~data['review/text'].isna()]
# Select subset of data
data_subset = data[['beer/ABV', 'beer/beerId', 'beer/brewerId', 'beer/name', 'beer/style', 'review/overall', 'review/text']]
text_data  =data['review/text']
#%%
test_data = pd.read_csv('/Users/Hengyu/Desktop/Beer_Data/beer-ratings/test.csv')
test_data = test_data[~test_data['review/text'].isna()]
test_data_subset = test_data[['beer/ABV', 'beer/beerId', 'beer/brewerId', 'beer/name', 'beer/style', 'review/overall', 'review/text']]
test_text_data  =test_data['review/text']
#%%

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
stop_words = stopwords.words('english')
ps = PorterStemmer()
lm = WordNetLemmatizer()

#%%
# data cleaning process
# Returned bigrams in (x, x)
def data_tokenizer(sample_text):
    #lower case, removal punctuation, tokenize
    tokens = word_tokenize(sample_text.lower().translate(str.maketrans('', '', string.punctuation)))
    #stemming
    #tokens = [lm.lemmatize(token, pos = 'v') for token in tokens]
    tokens = [ps.stem(token) for token in tokens]
    bi_tokens = list(bigrams(tokens))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens, bi_tokens

def data_cleaning(sample_text):
    cleaned = word_tokenize(sample_text.lower().translate(str.maketrans('', '', string.punctuation)))
    tokens = [ps.stem(token) for token in cleaned]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def beer_dict_building(tokens_list):
    token_dict = {}
    for tokens in tokens_list:
        for token in tokens:
            if token in token_dict:
                token_dict[token] +=1
            else:
                token_dict[token] = 1
    return token_dict

def training_preparation(dataset, test_size, feature_col, target_col):
    train_data, test_data = train_test_split(dataset, test_size = test_size)
    y_train = train_data[target_col]
    y_test = test_data[target_col]
    X_train = train_data[feature_col]
    X_test = test_data[feature_col]

    tf = TfidfVectorizer(ngram_range = (1,2)).fit(X_train)
    X_train = tf.transform(X_train)
    X_test = tf.transform(X_test)
    return X_train, y_train, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred).reshape(1,-1)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Test run: cleaned_texts = text_data.apply(data_cleaning)
#%%
# Vanilla Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#%%
# Transformer binning the result into 3 categories
bins = [0,3,4,5]
data_subset['preference'] = pd.cut(data_subset['review/overall'], bins, include_lowest = True)
data_subset['target'] = LabelEncoder().fit_transform(data_subset['preference'])
data_subset['cleaned_text'] = data_subset['review/text'].apply(data_cleaning)
# Clean texts from test data
test_data_subset['cleaned_text'] = test_data_subset['review/text'].apply(data_cleaning)

#%%
# Save the data for later process
pickle.dump(data_subset, open('/Users/Hengyu/Desktop/Beer_Data/beer-ratings/processed_train.pickle', 'wb'))
pickle.dump(test_data_subset, open('/Users/Hengyu/Desktop/Beer_Data/beer-ratings/processed_test.pickle', 'wb'))
#%%
# Given that there is no overall data records for test, will split the train
# Training preparation
train_data, test_data = train_test_split(data_subset, test_size = 0.2)
y_train = train_data['target']
y_test = test_data['target']
X_train = train_data['cleaned_text']
X_test = test_data['cleaned_text']

tf = TfidfVectorizer(ngram_range = ((1,2))).fit(X_train)
X_train = tf.transform(X_train)
X_test = tf.transform(X_test)

#%%
baseline_model = LogisticRegression(penalty = 'l2', multi_class= 'ovr')
baseline_model.fit(X_train, y_train)
baseline_model.score(X_test, y_test)
y_pred = baseline_model.predict(X_test)
#%%
# Plotting confusion matrix
class_names = np.array(['Negative','Neutral','Positive'])
plot_confusion_matrix(y_test.tolist(), y_pred.tolist(), classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
# Seems like model is doing the best in neutral, which is not good given that netural training examples
# are taking the majority, hence the model might be making a lot of prediction of neutral. 

#%%
#Subsample the medium and positive preferences
neg = data_subset[data_subset['target'] == 0]
pos = data_subset[data_subset['target'] == 2]
neutral = data_subset[data_subset['target'] == 1]
neutral_sample = neutral.sample(n= neg.shape[0])
pos_sample = pos.sample(n = neg.shape[0])
sampled_data = pd.concat([neg, pos_sample, neutral_sample])
#%% 
# Data preparation and re-train the model
X_train, y_train, X_test, y_test = training_preparation(sampled_data, 0.2, feature_cols = ['cleaned_text'], target_col = 'target')
baseline_model = LogisticRegression(penalty = 'l2', multi_class= 'ovr')
baseline_model.fit(X_train, y_train)
baseline_model.score(X_test, y_test)

#%%
y_pred = baseline_model.predict(X_test)
class_names = np.array(['Negative','Neutral','Positive'])
plot_confusion_matrix(y_test.tolist(), y_pred.tolist(), classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

## Compared with previous result, it is doing way better in predicting 
## negative and positive, with similar accuracy.


#%%
