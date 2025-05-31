import pandas as pd

# Load dataset
df = pd.read_csv('/Users/andrew/Downloads/labeled_data.csv',index_col=0)

# View first 5 rows of dataframe
df.head()


# The dataset has 6 columns. The 'tweet' column contains the tweets. Class has three unique values - 0,1 and 2. 
# 
# 0 - hate speech
# 
# 1 - offensive language
# 
# 2 - neither
# 
# The 'hate_speech', 'offensive_language' and 'neither' columns contain corresponding ratings of individual tweets. The 'count' column contains the count of total ratings for a tweet. 

# Looking at a single tweet to get an idea of what to expect 
df['tweet'][5]

# Looking at metadata of the dataframe 
df.info()

# From the above, I can see that the dataset has 24783 rows and 6 columns. 5 of the columns contain int64 data

# Creating a new feature called tweet_length
df['tweet_length'] = df['tweet'].apply(len)

# Checking that last action did not introduce an error
df.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Visualing average tweet length by class
sns.barplot(df,x='class',y='tweet_length')
plt.title('Average tweet length by class')
plt.show()

# Distribution of tweet length by class
sns.histplot(data=df[df['tweet_length']<=280],
             x=df[df['tweet_length']<=280]['tweet_length'],
             hue=df[df['tweet_length']<=280]['class'],         
             color='blue',
             bins=30)
plt.title('Distribution by tweet length')
plt.show()

# Number of tweets by class
df.groupby('class').count()['tweet']

# Maximum tweet length
max_index = df['tweet_length'].idxmax()
max_index

# Displaying longest tweet
# Considering that the max length of a tweet is 280, this looks like an error. Consulting stakeholders to confirm 
# why tweets like this exist in our data before dropping them is a good idea. 

df[df['tweet_length'] == df['tweet_length'].max()]['tweet'].loc[max_index]

# Tweets with more than 280 characteristics
df[df['tweet_length'] >= 280]['tweet'].info()

# Creating a new dataframe without the rows with tweets with more than 280 characters
df_v2 = df[df['tweet_length'] <= 280]

df_v2.info()

# Displaying first five rows of new dataset
df_v2.head()

# Distribution of tweet_length by class
import matplotlib.pyplot as plt
df_v2.hist(column='tweet_length',by='class',bins=20)
plt.show()

# Mean tweet length by class
df_v2.groupby('class')['tweet_length'].mean()

def remove_username(tweet):
    '''Function that takes a tweet as an agreement and removes 
    twitter handle
    '''
    words_list = []
    for word in tweet.split():
        if not word.startswith('@'):
            words_list.append(word)
    return ' '.join(words_list)

# Libraries needed for custom text_processing function below
import string
from nltk.corpus import stopwords

def text_processing(tweet):
    ''' Function takes in tweet as argument and returns a list of 
    of words in the string with stopwords, punctuation marks and 
    the twitter handle removed'''
    words_list = []
    for word in tweet.split():
        if word.startswith('@'):
            continue  # Skip twitter handles
        else:
            words_list.append(word)

    nopunc = [char for char in ' '.join(words_list) if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#df_v2['tweet_edited'] = df_v2['tweet'].apply(remove_username)

# New tweet feature created using text_processing function
df_v2['tweet_edited'] = df_v2['tweet'].apply(text_processing)

df_v2.head()

df_v2['hate_speech_rating'] = df_v2['hate_speech']/df_v2['count']
df_v2['offensive_language_rating'] = df_v2['offensive_language']/df_v2['count']
df_v2['neither_rating'] = df_v2['neither']/df_v2['count']

df_v2.head(2)

# Library that tranforms numerals into words
from num2words import num2words

num2words(1.33)

# Transforming numeric ratings into words

df_v2['hate_speech_rating_words'] = df_v2['hate_speech_rating'].apply(lambda x: num2words(round(x,1)))

df_v2['offensive_language_rating_words'] = df_v2['offensive_language_rating'].apply(lambda x: num2words(round(x,1)))

df_v2['neither_rating_words'] = df_v2['neither_rating'].apply(lambda x: num2words(round(x,1)))

df_v2.head(2)

# Transforming list of words in tweet_edited column to sentences
df_v2['tweet_edited'] = df_v2['tweet_edited'].apply(lambda x: ' '.join(x))

# Combining tweet_edited and rating in words
df_v2['tweet_edited_with_ratings'] = df_v2['tweet_edited'] + ' ' + df_v2['hate_speech_rating_words'] + ' ' + df_v2['offensive_language_rating_words'] + ' ' + df_v2['neither_rating_words']

# Checking that it worked
df_v2['tweet_edited_with_ratings'][9]

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Building models to classify the 3 groups

tweet_train,tweet_test,label_train,label_test=train_test_split(df_v2['tweet_edited_with_ratings'],
                                                           df_v2['class'],
                                                           test_size=0.2,
                                                              stratify=df_v2['class'])
tweet_train.head(2)

# ### Naive Bayes Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(tweet_train,label_train)

predictions = pipeline.predict(tweet_test)


print(confusion_matrix(label_test, predictions))
print('\n')
print(classification_report(label_test, predictions)) 


# Decision Tree Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])

pipeline.fit(tweet_train,label_train)

predictions = pipeline.predict(tweet_test)


print(confusion_matrix(label_test, predictions))
print('\n')
print(classification_report(label_test, predictions))      


# Random Forest Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])
pipeline.fit(tweet_train,label_train)

predictions = pipeline.predict(tweet_test)

print(confusion_matrix(label_test, predictions))
print('\n')
print(classification_report(label_test, predictions))      


# K Nearest neighbors

from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', KNeighborsClassifier(n_neighbors=7))
])

pipeline.fit(tweet_train,label_train)

predictions = pipeline.predict(tweet_test)


print(confusion_matrix(label_test, predictions))
print('\n')
print(classification_report(label_test, predictions)) 

# Support Vector Machine Algorithm

from sklearn.svm import SVC

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', SVC())
])

pipeline.fit(tweet_train,label_train)

predictions = pipeline.predict(tweet_test)


print(confusion_matrix(label_test, predictions))
print('\n')
print(classification_report(label_test, predictions))   


# K Nearest Neighbors Classifier

# Bag of words used in all tweets are created
# CountVectorizer class is used to transform the words into numeric values
bow_transformer = CountVectorizer().fit(df_v2['tweet_edited_with_ratings'])

tweets_bow = bow_transformer.transform(df_v2['tweet_edited_with_ratings'])

# Showing total number of words used 
print(len(bow_transformer.vocabulary_))

# The code below can be used to show the words used and the numeric value assigned to them
# Output is a dictionary
# print(bow_transformer.vocabulary_)

from sklearn.feature_extraction.text import TfidfTransformer

# Tfidf counts the number of times a word appears and attaches a weight to it
# The result is a sparse matrix
tfidf_transformer = TfidfTransformer().fit(tweets_bow)

# Transforming one row before transforming the entire column to confirm the result
tweet5 = df_v2['tweet_edited_with_ratings'][5]
print(tweet5)

bow5 = bow_transformer.transform([tweet5])
print(bow5)
print(bow5.shape)

tfidf5 = tfidf_transformer.transform(bow5)
print(tfidf5)

# Tranforming the entire column and displaying the shape and data type
tweets_tfidf = tfidf_transformer.transform(tweets_bow)
print(tweets_tfidf.shape)
type(tweets_tfidf)

tweet_train,tweet_test,label_train,label_test=train_test_split(tweets_tfidf,df_v2['class'],test_size=0.3,stratify=df_v2['class'])

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Using elbow method to find the right k value

error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(tweet_train,label_train)
    pred_i = knn.predict(tweet_test)
    error_rate.append(np.mean(pred_i != label_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# NOW WITH K=7
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(tweet_train,label_train)
pred = knn.predict(tweet_test)

print('WITH K=7')
print('\n')
print(confusion_matrix(label_test,pred))
print('\n')
print(classification_report(label_test,pred))


# ## Two model approach

# Two models are built:
# 
# Model 1 - To classify 'okay language' and 'bad language'
# 
# Model 2 - To classify 'bad language' into 2 groups; offensive language and hate speech

df_v3 = df_v2.copy()

df_v3.head(2)


# ### Model One

# ### New classes created: 
# #### 0 - neither (okay language)
# #### 1 - offensive language and hate speech (bad language)

# In[64]:


df_v3['class_new'] = df_v3['class'].apply(lambda x: 0 if x == 1 else 1)

df_v3['class_new'].unique()

df_v3['class'].value_counts().plot(kind='bar')
plt.title('Original class imbalance')
plt.show()

df_v3['class_new'].value_counts().plot(kind='bar')
plt.title('New class imbalance')
plt.show()

# New class imbalance
class_1 = round(df_v3['class_new'].value_counts()[1] / len(df_v3),2)
[1-class_1, class_1]


# ### Naive Bayes Classifier

tweet__train,tweet__test,label__train,label__test=train_test_split(df_v3['tweet_edited_with_ratings'],
                                                           df_v3['class_new'],
                                                           test_size=0.2,
                                                              stratify=df_v3['class_new'])

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', BernoulliNB())
])

pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)


print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions)) 


# ### DecisionTree classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])

pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)


print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions))      


# ### Random Forest Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])
pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)

print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions))      


# Support Vector Machines Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', SVC())
])

pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)


print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions))   


# K Nearest Neighbors

tweet_train,tweet_test,label_train,label_test=train_test_split(tweets_tfidf,df_v3['class_new'],test_size=0.3,stratify=df_v3['class_new'])

error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(tweet_train,label_train)
    pred_i = knn.predict(tweet_test)
    error_rate.append(np.mean(pred_i != label_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


# Set the k value to the value determined using the elbow method
# Let's assume k = 5

# NOW WITH K=5
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(tweet_train,label_train)
pred = knn.predict(tweet_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(label_test,pred))
print('\n')
print(classification_report(label_test,pred))


# Model Two
# Classes are:
# 0 - Offensive language 
# 1 - Hateful speech


# Filtering dataset to include only offensive language and hateful speech rows
df_v4 = df_v2[(df_v2['class'] == 0) | (df_v2['class'] == 2)]

df_v4.info()

df_v4.head(2)

# Further filtering to include only relevant columns
df_v4_filtered = df_v4[['tweet_edited_with_ratings','class']]

# Renaming classes
# 0 - offensive language
# 1 - hateful speech
df_v4_filtered['class_new'] = df_v4_filtered['class'].apply(lambda x: 0 if x == 2 else 1)

df_v4_filtered.head(10)

tweet__train,tweet__test,label__train,label__test=train_test_split(df_v4_filtered['tweet_edited_with_ratings'],
                                                           df_v4_filtered['class_new'],
                                                           test_size=0.25,
                                                              stratify=df_v4_filtered['class_new'])


# Naive Bayes Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', BernoulliNB())
])

pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)


print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions)) 


# Decision Tree Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', DecisionTreeClassifier())
])

pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)


print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions))      


# Random Forest Classifier

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)


print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions))   


# Support Vector Machines

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier', SVC())
])

pipeline.fit(tweet__train,label__train)

predictions = pipeline.predict(tweet__test)


print(confusion_matrix(label__test, predictions))
print('\n')
print(classification_report(label__test, predictions))   


# The one model approach to classify the three groups does not yield good results.
# The champion of the algorithms used is the Decision tree classifier.
# While the model predicts the 'neither' class and 'offensive language' class quite well,
# it does not do well with the minority class - 'hateful speech'

# The two model approach yields better results.
# The champion is the Support Vector Machines algorithm for both model 1 and model 2



