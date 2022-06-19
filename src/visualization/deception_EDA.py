# Removing warnings
import warnings

warnings.filterwarnings('ignore')

# Data Munging and Numerical Computing
import pandas as pd
from numpy import mean, median

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the readability metric for scoring the comment/text complexity
import readability

#  Convokit Corpus
from convokit import Corpus, download

# Import Utilities class for data cleaning
from NLUProject.src.utilities.functions import *

# Initializing the layout for plots
sns.set_context("paper", font_scale=1, rc={"grid.linewidth": 3})
pd.set_option('display.max_rows', 100, 'display.max_columns', 400)

# Download the deception CONVOKIT corpus
corpus = Corpus(filename=download("diplomacy-corpus"))

# Quick Stats
print(corpus.print_summary_stats())

# Getting dataset utterances
data = corpus.get_utterances_dataframe()
print(data.head())

# Just an example text
print(clean("I am a #Starkid"))

# Perform Data Cleaning
data['clean_text'] = data['text'].apply(lambda text: clean(text))

# Compute readability metrics of the data
data['readability_kincaid'] = readability.getmeasures(data['clean_text'], lang='en')['readability grades']['Kincaid']
data['readability_GF'] = readability.getmeasures(data['clean_text'], lang='en')['readability grades']['GunningFogIndex']

# Compute average Kincaid Grade level
print(f"Average Kincaid Grade Level of document is %0.2f" % (data['readability_kincaid'].mean()))

# Compute average Gunning Fog Index Level
print(f"Average Gunning Fog Index of document is %0.2f" % (data['readability_GF'].mean()))

# Source - https://www.wyliecomm.com/2021/11/measure-reading-levels-with-readability-indexes/
# The value of Kincaid Grade level of 6.1 and GF Index around 10.6 means that the text is EASILY READABLE by the common reader
# These values suggest that the overall readability of the text is permissible for a sophomore. Hence,  it can be understood by most people

# Computing these scores where there is an occurrence of deception
data_decept = data[data['meta.speaker_intention'] == 'Lie']
data_decept = data_decept[data_decept['meta.receiver_perception'] == 'Truth']

# # Compute readability metrics of the new data
print("Kincaid Readability Level of deceptive text is %0.2f" % (data_decept['readability_kincaid'].mean()))
print("Gunning Fog Index Readability Level of deceptive text is %0.2f" % data_decept['readability_GF'].mean())

# Compute average text length and compare with the length of deceptive text
data['sent_length'] = data['clean_text'].apply(lambda text: len(text))
data_decept['sent_length'] = data_decept['clean_text'].apply(lambda text: len(text))

# Print average lengths
print("Mean average length of data %0.2f" % (data['sent_length'].mean()))
print(data_decept['sent_length'].mean())

# Deceptive texts are usually longer as they consist of more positive connotations, phrases and clauses than normal text.
# Also, it consists of more verbose to convince or appease the listener

data['readability_kincaid'].value_counts()

# Study the distribution of deception quadrant
sns.displot(data.dropna(axis=0, how='all'), x='meta.deception_quadrant', hue='meta.deception_quadrant')
plt.show()

# EDA for all games
# Message Count
print(f"Message Count - {data.shape[0]}")

# Actual Lie Count
data_act_lie = data[(data['meta.speaker_intention'] == 'Lie')]
print(f"Actual Lie Count - {data_act_lie.shape[0]}")

# Suspected Lie Count
data_sus = data[data['meta.speaker_intention'] == 'Truth']
data_sus_lie = data_sus[data_sus['meta.receiver_perception'] == 'Lie']
print(f"Suspected Lie Count - {data_sus_lie.shape[0]}")

# Average Word Count
data_length = data['clean_text'].apply(lambda sent: len(sent))
print(f"Average Word Count - {(data_length.values.sum()) / len(data):.2f}")

# A lie-lie statement
data_caught = data_act_lie[data_act_lie['meta.receiver_perception'] == 'Lie']
data_caught['text'][0]

# Distribution of Word Count per message
data_length.plot.hist(grid=False)

# Word Count of Text by Speaker's perception
data['Word_Count'] = data['clean_text'].apply(lambda sent: len(sent))
plt.figure(figsize=(10, 15))
sns.displot(data.dropna(axis=0, how='all'), x='Word_Count', hue='meta.speaker_intention', multiple='stack',
            bins=[i for i in range(0, 2000, 100)])
plt.show()

# Word Count of Text by Receiver's perception
data['Word_Count'] = data['clean_text'].apply(lambda sent: len(sent))
plt.figure(figsize=(10, 15))
sns.displot(data.dropna(axis=0, how='all'), x='Word_Count', hue='meta.receiver_perception', multiple='stack',
            bins=[i for i in range(0, 2000, 100)])
plt.show()

# Calculating the mean AFINN sentiment score across all conversations
afinn_mean_conv_scores, emoji_count_convs = [], []
for conv_id in data['conversation_id'].unique():
    utterances, emoji_counts = [], []
    data_sample = data[data['conversation_id'] == conv_id]

    for utterance in data_sample['clean_text']:
        emoji_counts.append(count_emojis(utterance))

    # afinn_mean_conv_scores.append(afinn_sentiment_score(utterances))
    emoji_count_convs.append(emoji_counts)

print(f"The mean emoji count across all conversations in the dataset is %.4f" % mean(mean(emoji_count_convs)))

# Since the mean count of emojis is pretty low, it is not a good indicator of deceptive text
