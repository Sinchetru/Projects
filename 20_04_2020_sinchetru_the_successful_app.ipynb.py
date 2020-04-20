#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')
get_ipython().run_line_magic('pycodestyle_on', '')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Looking for your most successful application

# The purpose of the given project is the analysis of the market data and the elaboration of suggestions regarding the characteristics of mobile app profiles that are profitable for the App Store and Google Play markets and that will likely attract more users. Our aim is to enable developer's team to make data-driven decisions with respect to the kind of apps they build. The apps should be free to download and install, and the main source of revenue consists of in-app ads.
# 
# We will analyze existing data about applications developped for Android and iOS platform to determine the characteristics of an application that will attract a large number of users. To minimize risks and overhead, our validation strategy for an app idea is comprised of three steps:
# 
# * Build a minimal Android version of the app, and add it to Google Play.
# * If the app has a good response from users, we then develop it further.
# * If the app is profitable after six months, we also build an iOS version of the app and add it to the App Store.
# 
# Because our end goal is to add the app on both the App Store and Google Play, we need to find app profiles that are successful on both markets.
# ### Summary of Results
# 
# In this project, we analyzed survey data from Android Market and AppStore to find  the characteristics of an application that may be successful on the both market. 
# The only conclusion we reached is that the one of the free niches in which the application could fit is the category - Books and dictionaries. Considering the number of applications of this type and their popularity, a bilingual or trilingual English dictionary might be the right choice. 
# As an alternative, the development of an application of the Quran may be proposed.

# ### 1. Opening and Exploring the Data
# As of September 2018, there were approximately 2 million iOS apps available on the App Store, and 2.1 million Android apps on Google Play. To avoid spending money on organizing a survey, we'll first try to make use of existing data to determine whether we can reach any reliable result. Two data sets that seem suitable for our purpose were found on the www.kaggle.com web page.
# 
# - A data set containing data about approximately ten thousand Android apps from Google Play. 
# - A data set containing data about approximately seven thousand iOS apps from the App Store.

# In[2]:


# Importing packages that we will use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import seaborn as sns

# The Google Play and Apple Store data sets
android = 'C:/Users/Sinch/Desktop/Guided project/googleplaystore.csv'
ios = 'C:/Users/Sinch/Desktop/Guided project/AppleStore.csv'

df_android = pd.read_csv(android)
df_ios = pd.read_csv(ios)

df_android = df_android.copy()
df_ios = df_ios.copy()


# In[3]:


df_android.info()


# Column header description:
# * App - Application name
# * Category - Category the app belongs to
# * Rating - Overall user rating of the app (as when scraped)
# * Reviews - Number of user reviews for the app (as when scraped)
# * Size - Size of the app (as when scraped)
# * Installs - Number of user downloads/installs for the app (as when scraped)
# * Type - Paid or Free
# * Price - Price of the app (as when scraped)
# * Content Rating - Age group the app is targeted at - Children / Mature 21+ / Adult
# * Genres - An app can belong to multiple genres (apart from its main category). For eg, a musical family game will belong to Music, Game, Family genres.
# * Last Updated - Date when the app was last updated on Play Store (as when scraped)
# * Current Ver - Current version of the app available on Play Store (as when scraped)
# * Android Ver - Min required Android version (as when scraped)
# 
# 
# At a quick glance, the columns that might be useful for the purpose of our analysis are 'App', 'Size', 'Category', 'Reviews', 'Installs', 'Type', 'Price', and 'Genres'.

# In[4]:


df_ios.info()


# Column header description:
# * "id" : App ID
# * "track_name": App Name
# * "size_bytes": Size (in Bytes)
# * "currency": Currency Type
# * "price": Price amount
# * "rating_count_tot": User Rating counts (for all version)
# * "rating_count_ver": User Rating counts (for current version)
# * "user_rating" : Average User Rating value (for all version)
# * "user_rating_ver": Average User Rating value (for current version)
# * "ver" : Latest version code
# * "cont_rating": Content Rating
# * "prime_genre": Primary Genre
# * "sup_devices.num": Number of supporting devices
# * "ipadSc_urls.num": Number of screenshots showed for display
# * "lang.num": Number of supported languages
# * "vpp_lic": Vpp Device Based Licensing Enabled
# 
# Columns that may be useful for the purpose of our analysis are 'track_name', 'size_bytes', 'rating_count_tot', 'price', and 'prime_genre'.

# ### 2. Cleaning and Preparing the Data

# #### 2.1 Cleaning Android Data

# In[5]:


# Drop the columns we won't use
df_android = df_android.drop(columns=['Android Ver',
                                      'Current Ver',
                                      'Last Updated'])


# In[6]:


df_android.Rating.describe()


# The rating values cannot be more than 5, we will delete the rows with unrealistic data

# In[7]:


# Get names of indexes for which column Rating has value > 5
index_names = df_android[df_android['Rating'] > 5].index

# Delete these row indexes from dataFrame
df_android.drop(index_names, inplace=True)


# If we explore the Google Play data set long enough, we'll find that some apps have more than one entry.

# In[8]:


# Search for duplicates making a bool series
bool_series = df_android["App"].duplicated()

# display data
df_android[bool_series]


# In[9]:


# We have 1181 cases of dublicate values let's see some of them
df_android[df_android['App'] == 'Instagram']


# We don't want to count certain apps more than once when we analyze data, so we need to remove the duplicate entries and keep only one entry per app. One thing we could do is remove the duplicate rows randomly, but we could probably find a better way.
# 
# If you examine the rows we printed two cells above for the Instagram app, the main difference happens on the fourth position of each row, which corresponds to the number of reviews. The different numbers show that the data was collected at different times. We can use this to build a criterion for keeping rows. We won't remove rows randomly, but rather we'll keep the rows that have the highest number of reviews because the higher the number of reviews, the more reliable the ratings.

# In[10]:


max_review = df_android.groupby(['App']).Reviews.transform(max)
df_android = df_android.loc[df_android.Reviews == max_review]

# There are however some duplicates with the same max number of reviews, we will also drop them
df_android.drop_duplicates(subset=['App'], inplace=True)


# At the end we will check and fix the data types for a clean analysis:

# In[11]:


df_android.dtypes


# In[12]:


# Fixing Install values and turning them in to the float type
df_android['Installs'] = df_android['Installs'].str.replace(',', '')
df_android['Installs'] = df_android['Installs'].str.replace('+', '')
df_android['Installs'] = df_android['Installs'].astype(float)


# In[13]:


# Fixing Size values and turning them in to the float type
df_android['Size'] = df_android['Size'].str.replace(',', '')
df_android['Size'] = df_android['Size'].str.replace('M', '')


# We have both float and string values in the column and we will will try to leave the string in string format, for this reason we will convert only the numbers:

# In[14]:


def clean_float(column):
    for value in column:
        try:
            float(value)
        except ValueError:
            pass


clean_float(df_android['Size'])


# #### 2.2 Cleaning iOS Data

# In[15]:


df_ios.info()


# In[16]:


# Transforming the size column from bytes to MB
df_ios['size_Mbytes'] = df_ios['size_bytes'].div(1000000).round(2)


# In[17]:


# Drop the columns we won't use
df_ios = df_ios.drop(columns=['id',
                              'currency',
                              'size_bytes',
                              'rating_count_ver',
                              'user_rating_ver',
                              'ver',
                              'cont_rating',
                              'sup_devices.num',
                              'ipadSc_urls.num',
                              'lang.num',
                              'vpp_lic'])


# We will count the iOS app users exploring the 'rating_count_tot' column, for this reason we are not interested in the apps those have the rating count value equal to 0:

# In[18]:


# Selecting apps with 'rating_count_tot' > 0
df_ios = df_ios.drop(df_ios[df_ios['rating_count_tot'] == 0].index)


# #### 2.3 Cleaning non ASCII values

# In[19]:


df_ios.tail()


# Exploring the data we have noticed that the names of some of the apps suggest they are not directed toward an English-speaking audience. Above, we see a couple of examples from both data sets.
# 
# We're not interested in keeping these kind of apps, so we'll remove them. One way to go about this is to remove each app whose name contains a symbol that is not commonly used in English text — English text usually includes letters from the English alphabet, numbers composed of digits from 0 to 9, punctuation marks (., !, ?, ;, etc.), and other symbols (+, *, /, etc.).
# 
# All these characters that are specific to English texts are encoded using the ASCII standard. Each ASCII character has a corresponding number between 0 and 127 associated with it, and we can take advantage of that to build a function that checks an app name and tells us whether it contains non-ASCII characters.
# 
# Some English app names use emojis or other symbols (™, — (em dash), – (en dash), etc.) that fall outside of the ASCII range. We will check only the values that contains more as 3 non ASCII characters.

# In[20]:


def is_english(string):
    non_ascii = 0

    for character in string:
        if ord(character) > 127:
            non_ascii += 1

    if non_ascii > 3:
        return np.NaN
    else:
        return string


# Checking the Android and iOS Data for non ASCII app names:

# In[21]:


df_ios['track_name'] = df_ios.track_name.apply(is_english)
df_android['App'] = df_android.App.apply(is_english)


# In[22]:


# We have 1014 non ASCII values in iOS Dataset and 45 in Android Dataset
df_ios.isnull().sum()


# In[23]:


df_android.isnull().sum()


# In[24]:


df_ios = df_ios.dropna(subset=['track_name'])
df_android = df_android.dropna(subset=['App'])


# In[25]:


df_ios.info()


# In[26]:


df_android.info()


# In[27]:


df_ios.user_rating.describe()


# Our goal are the free apps for this reason we will delete all the non free apps rows

# In[28]:


df_android.Type.describe()


# We are interested only in free apps:

# In[29]:


# Selecting free Android apps
df_android = df_android.drop(df_android[df_android['Type'] != 'Free'].index)
# Selecting free iOS apps
df_ios = df_ios.drop(df_ios[df_ios['price'] != 0.0].index)

df_android.info()
df_ios.info()


# In[30]:


df_android


# In[31]:


df_ios


# ### 3. Analysing Data

# Because our end goal is to add the app on both the App Store and Google Play, we need to find app profiles that are successful on both markets. For instance, a profile that might work well for both markets might be a productivity app that makes use of gamification.
# We will begin the analysis by getting a sense of the most common genres for each market. For this, we'll build a frequency table for the *prime_genre* column of the App Store data set, and the Genres and *Category* columns of the Google Play data set.

# In[32]:


android_pivot = pd.pivot_table(df_android,
                               index=['Category'],
                               values=['App'],
                               aggfunc='count')

percent_and = android_pivot.sort_values('App', ascending=False).reset_index()
percent_and['Percentage'] = percent_and['App'] / percent_and['App'].sum()
percent_and['Percentage'] = (percent_and['Percentage'] * 100).round(2)
percent_and = percent_and.rename(columns={'App': 'Total'})

percent_and.head()


# In[33]:


ios_pivot = pd.pivot_table(df_ios,
                           index=['prime_genre'],
                           values=['track_name'],
                           aggfunc='count')

percent_ios = ios_pivot.sort_values('track_name', ascending=False).reset_index()
percent_ios['Percentage'] = percent_ios['track_name'] / percent_ios['track_name'].sum()
percent_ios['Percentage'] = (percent_ios['Percentage'] * 100).round(2)
percent_ios = percent_ios.rename(columns={'track_name': 'Total',
                                          'prime_genre': 'Category'})

percent_ios.head()


# In[34]:


sns.set(style="whitegrid")

figure = plt.figure(2, figsize=(20, 8))
the_grid = figure.add_gridspec(2, 2)

plt.subplot(the_grid[0, 1],  title='Android app ')
sns.barplot(x='Percentage',
            y='Category',
            data=percent_and.head(7),
            palette='Spectral')

plt.subplot(the_grid[1, -1], title='iOS app')
sns.barplot(x='Percentage',
            y='Category',
            data=percent_ios.head(7),
            palette='Spectral')

plt.suptitle('The repartition of applications by category',
             x=0.78,
             y=1.05,
             fontsize=16)

plt.tight_layout()


# For the App Store we can notice that among the free English apps, a good part of them (58.16%) are games. Entertainment apps are close to 8%, followed by photo and video apps, which are close to 5%. Only 3.66% of the apps are designed for education, followed by social networking apps which amount for 3.29% of the apps in our data set.
# 
# The situation on Google Play is completely different - there are not that many apps designed for fun, and it seems that a good number of apps are designed for practical purposes (family, tools, business, lifestyle, productivity, etc.). However the market is dominated (29%) by Family (the biggest part of this category consist from games for kids also actually there are more kid's games as family apps) and Games categories.
#  
# * The English speaking App Store is dominated by apps that are designed for fun (games, entertainment, photo and video, social networking, sports, music, etc.), while apps with practical purposes (education, shopping, utilities, productivity, lifestyle, etc.) are more rare. However, the great number of fun apps those are the most numerous doesn't mean that they are the most popular, we will try to check this supposition later.
# 
# * The situation on Google Play is more balanced, the market being dominated by applications such as family and games, but besides these almost 17% of the market are occupied by applications such as Tools, Business and Lifestyle and this niche can be very interesting.
# 
# 
# 

# #### 3.1 Most Popular Apps by Genre on Apple Store

# For the Google Play data set, we have used the information in the Installs column, but for the App Store data set this information is missing. As a workaround, we'll take the total number of user ratings as a proxy, which we can find in the *rating_count_tot* column

# In[35]:


ios_pivot = pd.pivot_table(df_ios, index=['prime_genre'],
                           values=['rating_count_tot'],
                           aggfunc=np.mean)

ios_pivot = ios_pivot.sort_values('rating_count_tot', ascending=False).head(10)

plt.figure(figsize=(16, 6))
sns.barplot(x=ios_pivot.index, y='rating_count_tot', data=ios_pivot)
plt.tight_layout()


# On average, navigation apps have the highest number of user reviews, but this figure is heavily influenced by Waze and Google Maps, which have close to half a million user reviews together. The same pattern applies to social networking apps, where the average number is heavily influenced by a few giants like Facebook, Pinterest, Skype, etc. Same applies to music apps, where a few big players like Pandora, Spotify, and Shazam heavily influence the average number.

# In[36]:


df_ios[df_ios['prime_genre'] == 'Navigation'].sort_values(by='rating_count_tot',
                                                          ascending=False).head(5)


# Reference apps have 86 090 user ratings on average, but it's actually the Bible and Dictionary.com which skew up the average rating:

# In[37]:


df_ios[df_ios['prime_genre'] == 'Reference'].sort_values(by='rating_count_tot',
                                                         ascending=False).head(10)


# However, this niche seems to show some potential and developing a Dictionary or a Bible App can be a good idea. This idea seems to fit well with the fact that the App Store is dominated by for-fun apps. This suggests the market might be a bit saturated with for-fun apps, which means a practical app might have more of a chance to stand out among the huge number of apps on the App Store.
# 
# Other genres that seem popular include weather, book, food and drink, or finance. The book genre seem to overlap a bit with the app idea we described above, but the other genres don't seem too interesting to us.
# 
# * Weather apps — people generally don't spend too much time in-app, and the chances of making profit from in-app adds are low. Also, getting reliable live weather data may require us to connect our apps to non-free APIs.
# * Food and drink — examples here include Starbucks, Dunkin' Donuts, McDonald's, etc. So making a popular food and drink app requires actual cooking and a delivery service, which is outside the scope of our company.
# * Finance apps — these apps involve banking, paying bills, money transfer, etc. Building a finance app requires domain knowledge, and we don't want to hire a finance expert just to build an app.

# #### 3.2 Most Popular Apps by Genre and Category on Google Play

# Let's have a look which are the most popular apps in a couple of categories, we will try to analyze the number of installations:

# In[38]:


android_pivot = pd.pivot_table(df_android, index=['Category'],
                               values=['Installs'], aggfunc=np.sum).head(15)

android_pivot = android_pivot.sort_values('Installs', ascending=False).head(8)

plt.figure(figsize=(16, 6))
sns.barplot(x=android_pivot.index, y='Installs', data=android_pivot)
plt.tight_layout()


# Let's dive inside the most popular categories:

# In[39]:


df_android[df_android['Category'] == 'FAMILY'].sort_values(by='Installs',
                                                           ascending=False).head(5)


# In[40]:


df_android[df_android['Category'] == 'GAME'].sort_values('Installs',
                                                         ascending=False).head(5)


# In[41]:


df_android[df_android['Category'] == 'COMMUNICATION'].sort_values('Installs',
                                                                  ascending=False).head(5)


# In[42]:


df_android[df_android['Category'] == 'BOOKS_AND_REFERENCE'].sort_values('Installs', ascending=False).head(5)


# 
# Let's try to analyze the picture more broadly, we will combine genres and categories:

# In[43]:


android_pivot = pd.pivot_table(df_android, index=['Category', 'Genres'],
                               values=['Installs'], aggfunc=np.sum).head(15)

android_pivot.sort_values('Installs', ascending=False)
android_pivot


# On average, communication apps are very popular. Theirs number is heavily skewed up by the giants like WhatsApp, Facebook, Messenger, Skype, Google Chrome, Gmail, and Hangouts that have over one billion installs.
# 
# The market is also dominated by apps like Youtube, Google Play Movies & TV, or MX Player. The pattern is repeated for social apps (where we have strong players like Facebook, Instagram, Google+, etc.), photography apps (Google Photos and other popular photo editors), or productivity apps (Microsoft Word, Dropbox, Google Calendar, Evernote, etc.). We will try to find a niche that it is not so dominated by strong players who are hard to compete against. The game genre seems pretty popular, but 29% of the total apps on the market seems a bit saturated, so we'd like to come up with a different app recommendation if possible.
# 
# The books and reference genre seems to be popular as well, with 1.665884e+09 installs. It can be an interesting niche for us, let's try to analyze it more deeply.

# The age category seems not very differentiate, that means that the future app should be developed for a non-age limited auditory.

# In[44]:


android_pivot = pd.pivot_table(df_android[df_android['Category'] == 'BOOKS_AND_REFERENCE'],
                               index=['Category', 'Content Rating'],
                               values=['Installs'],
                               aggfunc=np.sum)
android_pivot


# The book and reference genre includes a variety of apps: software for processing and reading ebooks, various collections of libraries, dictionaries, tutorials on programming or languages, etc. It seems there's still a small number of extremely popular apps that skew the average.
# However, it looks like there are only a few very popular apps, so this market still shows potential. Let's try to get some app ideas based on the kind of apps that are somewhere in the middle in terms of popularity with more than 500 000 downloads:

# In[45]:


df_android_cat = df_android[df_android['Category'] == 'BOOKS_AND_REFERENCE'].sort_values('Installs', ascending=False)
df_android_cat = df_android_cat[df_android_cat['Installs'] >= 500000]
df_android_cat


# Let's search among the 69 applications which kind of apps do we have there. This niche seems to be dominated by electronic readers and dictionaries, so it will be probably to compete developing a similar app. 

# In[46]:


import re
patterns = ['Dict', 'Book', 'Tutorial', 'Audio', 'Bible', 'Quran']

patterns_lower = '|'.join([s.lower() for s in patterns])
df_android_cat['App'].str.lower().str.extract(rf"({patterns_lower})",
                                              expand=False).value_counts()


# However we can notice that the mos of the dictionaries are English or English-Hindi. It seems that developing an English bilingual dictionary could be profitable for both the Google Play and the App Store markets.

# Let's have a look on the the TOP most important languages in the world in the 21st century:

# <img src='https://www.daytranslations.com/blog/wp-content/uploads/2017/07/What-Are-The-Most-Important-Languages-of-The-21st-Century.jpg' width='600' height='600' />

# As we see the most popular languages of the 21st century are English, Chinese, German and Spanish, therefore a bilingual English-German or trilingual English-German-Chinese dictionary could be a successful choice.
# 
# We also notice there are quite a few apps built around the book Quran, which suggests that building an app around a popular book can be profitable. It seems that taking a popular book (perhaps a more recent book) and turning it into an app could be profitable for both the Google Play and the App Store markets.
# 
# In order to diversify the developed application from the competing apps it is necessary to add some special features besides the raw version of the book. This might include daily quotes from the book, an audio version of the book, quizzes on the book, a forum where people can discuss the book, etc.

# ### Conclusion

# In this project, we analyzed survey data from Android Market and AppStore to find  the characteristics of an application that may be successful on the both market. 
# The only conclusion we reached is that the one of the free niches in which the application could fit is the category - Books and dictionaries. Considering the number of applications of this type and their popularity, a bilingual or trilingual English dictionary might be the right choice. 
# As an alternative, the development of an application of the Quran may be proposed.
