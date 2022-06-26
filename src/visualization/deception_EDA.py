# Import Standard Libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Import third-party libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import readability
from convokit import Corpus, download # Convokit Corpus


# Import utilities
from src.utilities.functions import clean_reddit


corpus = Corpus(filename=download("diplomacy-corpus")) # Download the DECEPTION IN DIPLOMACY corpus
data = corpus.get_utterances_dataframe() # Get UTTERANCES in the dataset
data['clean_text'] = data['text'].apply(lambda text: clean_reddit(text)) # Perform Data Cleaning

data['readability'] = readability.getmeasures(data['clean_text'], lang='en')['readability grades']['Kincaid'] # Compute readability metrics of the data
data['readability_GF'] = readability.getmeasures(data['clean_text'], lang='en')['readability grades']['GunningFogIndex']
data['sent_length'] = data['clean_text'].apply(lambda text: len(text))  # Compute average text length and compare with the length of deceptive text

sns.displot(data.dropna(axis=0, how='all'), x='meta.deception_quadrant', hue='meta.deception_quadrant') #Study the distribution of deception quadrant
plt.show()

data_deceive = data[data['meta.speaker_intention']=='Lie']  # Computing these scores where there is an occurence of deception
data_deceive = data_deceive[data_deceive['meta.receiver_perception']=='Truth']

data_act_lie = data[(data['meta.speaker_intention']=='Lie')]    # Actual Lie Count

data_sus = data[data['meta.speaker_intention'] == 'Truth']      # Suspected Lie Count
data_sus_lie = data_sus[data_sus['meta.receiver_perception'] == 'Lie']


data_length=data['text'].apply(lambda sent : len(sent))            # Average Word Count

data_caught = data_act_lie[data_act_lie['meta.receiver_perception']=='Lie'] # A lie-lie statement

data_length.plot.hist(grid=False)   # Distribution of Word Count per message

data['Word_Count'] = data['text'].apply(lambda sent : len(sent))    # Word Count of Text by Speaker's perception
plt.figure(figsize=(10, 15))
sns.displot(data.dropna(axis=0, how='all'), x='Word_Count', hue='meta.speaker_intention', multiple='stack', bins=[i for i in range(0, 2000, 100)])
plt.show()

data['Word_Count'] = data['text'].apply(lambda sent : len(sent))    # Word Count of Text by Receiver's perception
plt.figure(figsize=(10, 15))
sns.displot(data.dropna(axis=0, how='all'), x='Word_Count', hue='meta.receiver_perception', multiple='stack', bins=[i for i in range(0, 2000, 100)])
plt.show()

data_deceive = data_deceive.reset_index() # Resetting the index of dataframe
data_deceive['cnt_modifiers'] = data_deceive['clean_text'].apply(lambda text: count_mod(text))  # Capture Count of Modifiers for the entire dataframe sum them according to user values

# Capture modal verbs that indicate possibility 
# modals_by_id = {}
# data_decep['cnt_mods'] = data_decep['clean_text'].apply(lambda text: cnt_possib(text))
# for user_id in data_decep['user_id_vals'].unique():
#     data_user = data_decep[data_decep['user_id_vals'] == user_id]
#     modals_by_id[user_id] = sum(data_user['cnt_mods'])

    
# pp_by_id = {}
# data_decep
# data_decep['cnt_mods'] = data_decep['clean_text'].apply(lambda text: cnt_possib(text))
# for user_id in data_decep['user_id_vals'].unique():
#     data_user = data_decep[data_decep['user_id_vals'] == user_id]
#     modals_by_id[user_id] = sum(data_user['cnt_mods'])

self_by_id = {}
data_decep['self_ref'] = data_decep['clean_text'].apply(lambda text: cnt_self_ref(text))
for user_id in data_decep['user_id_vals'].unique():
    data_user = data_decep[data_decep['user_id_vals'] == user_id]
    self_by_id[user_id] = sum(data_user['self_ref'])
self_by_id = dict(sorted(self_by_id.items(), key = lambda x: x[1], reverse = True))

# Capture modifier count by user IDs
# mod_by_id = {}
# for user_id in data_decep['user_id_vals'].unique():
#     data_user = data_decep[data_decep['user_id_vals'] == user_id]
#     mod_by_id[user_id] = sum(data_user['cnt_modifiers'])
# data_decep['user_id_vals'] = data_decep['id'].map(decept_user_map)
# deceptive_users = list(data_decep['user_id_vals'].unique())


# for user_id in deceptive_users:
#     data_user = data_decep[data_decep['user_id_vals']==user_id]
#     = data_user['text'].apply(lambda text: count_mod(text))
#     mod_by_id[user_id] = count_mod
    
# Create a count and plot a histogram. Calculate average value and keep it as a threshold 

# Plot and save average value. Extend the average value to future datasets for verification 


# In[65]:


decept_user_map = pd.{user: val for (val, user) in enumerate(deceptive_users)}


# In[85]:


sum(data_decep['cnt_modifiers'])


# In[89]:


mod_by_id.values()


# In[94]:


# Test of modality 
modals_by_id = sorted()
plt.hist(modals_by_id.values())


# In[122]:


plt.hist(self_by_id.values())


# In[149]:


data_extra = data[data['meta.speaker_intention']=='Truth']
data_extra = data[data['meta.receiver_perception']=='Truth']
data_extra = data_extra.reset_index()

self_by_id_auth = {}
data_extra['self_ref'] = data_extra['clean_text'].apply(lambda text: cnt_self_ref(text))
for user_id in data_extra['id'].unique():
    data_user = data_extra[data_extra['id'] == user_id]
    self_by_id_auth[user_id] = sum(data_user['self_ref'])
    
self_by_id_auth = dict(sorted(self_by_id_auth.items(), key = lambda x: x[1], reverse = True))


# In[151]:


self_by_id_auth


# In[159]:


import random

samp1 = random.sample(self_by_id.keys(), k=500)
samp2 = random.sample(self_by_id_auth.keys(), k=500)
sample_deceive_dict = {key: self_by_id[key] for key in samp1}
sample_truth_dict = {key: self_by_id_auth[key] for key in samp2}


# In[161]:


plt.hist(sample_deceive_dict.values())


# In[162]:


plt.hist(sample_truth_dict.values())

