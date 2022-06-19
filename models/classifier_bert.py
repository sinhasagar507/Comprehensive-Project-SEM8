#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/mistryishan25/Comprehensive-Project-SEM8/blob/master/models/1_0_ish_Node_Classifier_Bert.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Resource dump:
# 1. Which Free GPUs - [Article](https://towardsdatascience.com/free-gpus-for-training-your-deep-learning-models-c1ce47863350)
# 2. Interpret the working of BERT - [Article](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)
# 3. Baseline : Naive Bayes + TF-IDF - [Article](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/) 

# In[ ]:


# !pip install wandb


# In[ ]:


# !pip install convokit


# In[ ]:


# !pip install contractions


# In[ ]:


# !pip install transformers


# ### Imports and citations 
# 
# Characterizing Online Discussion Using Coarse Discourse Sequences,
# Amy Zhang,Bryan Culbertson,Praveen Paritosh
# 
# 
# Bibtex
# @inproceedings{46055,
# title	= {Characterizing Online Discussion Using Coarse Discourse Sequences},
# author	= {Amy Zhang and Bryan Culbertson and Praveen Paritosh},
# year	= {2017}
# }
# 
# Reference : [Link](https://colab.research.google.com/github/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/Introduction_to_ConvoKit.ipynb#scrollTo=kHB78-JtViKt)

# In[72]:


# Data related imports
import convokit
from convokit import Corpus, download
import re
# import contractions

# Data manipulation
import pandas as pd

# utilities
# from google.colab import output


# In[ ]:


BASE_CORPUS = Corpus(download("reddit-coarse-discourse-corpus"));
#corpus.print_summary_stats();


# ### What each utterance contains?
# 1. comment_depth: depth of the comment, 0 if the utterance is the top-level post itself.
# 2. majority type: discourse action type by one of the following: question, answer, announcement, agreement, appreciation, disagreement, elaboration, humor
# 3. annotation_types (list of annotation types by three annotators)
# 4. majority_link : link in relation to previous post, none if no relation with previous comment
# 5. annotation_links (list of annotation links by three annotators)
# 

# In[8]:


utt_5 = []
for i in range(5):
    utt = BASE_CORPUS.random_utterance()
    utt_5.append(utt)
    print(utt.text)
    print("-"*40)
    # print(utt.meta)


# ### Explore the conversations
# 
# 1. [ ] How many emojis do we actually have? Are they used instead of words?
# 2. [ ] Explore the distribution of the labels. 

# - We would be needing only the text and the meta.
# - Vector attribute stays we might add the vector data for each utterance for other cases- [Link](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/vectors/vector_demo.ipynb)

# In[9]:


df = BASE_CORPUS.get_utterances_dataframe()
df.head()


# In[10]:


# Yes indeed these are the number of utterances in the entire dataset.
len(df)


# In[11]:


# Meh not important here anyways
df = df.drop(["timestamp"], axis=1)

# Vectors are also null in a way as they will updated later on with the choice of embedding in the metadata


# In[73]:


# Checking for missing values : 

def missing_values(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    # Seperating them into a new df for use later on while testing
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    return missing_value_df


# ### List of Cleaning to be done
# 1. [x] Clean URls - Convention is [text]\(URL)
# 2. [x] Do not seperate the quoted text - "> blah /n/n
# 3. [x] Removing special characters 
# 4. [ ] Speling correction? What about the ones that convey info?
# 5. [ ] Deal with Emojis and emoticons - [emot lib](https://medium.com/geekculture/text-preprocessing-how-to-handle-emoji-emoticon-641bbfa6e9e7) 
# 6. [x] Contractions need to be taken care of -
# 7. [x] Remove /n

# Using Regex to do simple cleaning based on symbols - [Documentation](https://docs.python.org/3/library/re.html#re.sub) and many stack-overflow pieces and articles

# In[13]:


def clean(text, newline=True, quote=True, bullet_point=True,dates=True,
          link=True, strikethrough=True, spoiler=True, heading=True, emoji=True, emoticon=True, contraction=True):
    
    # Newlines we dont need - only 
    if newline:
        text = re.sub(r'\n+', ' ', text)
        # Remove the many " " that we replaced in the last steo
        text = text.strip()
        text = re.sub(r'\s\s+', ' ', text)

    # > are for the qouted texts from the main comment or the reply
    if quote:
        text = re.sub(r'>', '', text)

    # Bullet points/asterisk are used for markdown like - bold/italic - Could create trouble in parsing? idk
    if bullet_point:
        text = re.sub(r'\*', '', text)
        text = re.sub('&amp;#x200B;', '', text)

    # []() Link format then we remove both the tag/placeholder and the link
    if link:
        text = re.sub(r"http\S+", '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # Strikethrough
    if strikethrough:
        text = re.sub('~', '', text)

    # Spoiler, which is used with < less-than (Preserves the text)
    if spoiler:
        text = re.sub('&lt;', '', text)
        text = re.sub(r'!(.*?)!', r'\1', text)

    # Heading to be removed as there are these markdown style features in reddit too
    if heading:
        text = re.sub('#', '', text)
        
    if emoji:
    # Implement the emoji scheme here. 
    # Makes more sense for the node feature but might as well import that function here if ready    
        pass
    if dates:
        text = re.sub(r'(\d+/\d+/\d+)', '', text)
    if emoticon:
    # Implement the emoticon scheme here. 
    # Makes more sense for the node feature but might as well import that function here if ready 
        pass
    
    #Needs to be the last step in the process
    # if contractions:
        # text = contractions.fix(text)
    #print("Running")    
    return text


# ### Ways of transformation to clean 
# 1.  Clean() then TextCleaner -----Meh
# 2. TextCleaner then Clean() ---------Meh 
# 3. Clean -------------------------MeH
# 4. TextCleaner ------------------Meh
# 5. Clean : Lambda ---------------Bingo

# The convo-kit has good interoperability with the sklearn package and hence we would have to exploit the sklearn stuff to do the transformation in a single pass instead of iterating over all the utterances - [CovoKit](https://convokit.cornell.edu/documentation/architecture.html#transformer)

# In[15]:


# # Option 1 - Clean -> TextCleaner
# corpus_1 = textCleaner.TextCleaner(text_cleaner= clean, input_field = "text").transform(BASE_CORPUS);
# corpus_2 = textCleaner.TextCleaner(input_field= "text").transform(corpus_1);
# df_1 = corpus_2.get_utterances_dataframe()


# In[16]:


# missing_1 = missing_values(df_1)
# missing_1


# In[17]:


# Option 2 TextCleaner -> Clean
# corpus_3 = textCleaner.TextCleaner(input_field= "text").transform(BASE_CORPUS)
# corpus_4 = textCleaner.TextCleaner(text_cleaner= clean, input_field = "text").transform(corpus_3)
# df_2 = corpus_4.get_utterances_dataframe()


# In[18]:


# missing_2 = missing_values(df_2)
# missing_2


# In[19]:


# Option 3 Clean
# corpus_5 = textCleaner.TextCleaner(text_cleaner= clean, input_field = "text").transform(BASE_CORPUS)
# df_3 = corpus_5.get_utterances_dataframe()


# In[20]:


# missing_3 = missing_values(df_3)
# missing_3


# In[21]:


# # Option 4 TextCleaner
# corpus_6 = textCleaner.TextCleaner(input_field= "text").transform(BASE_CORPUS)
# df_4 = corpus_6.get_utterances_dataframe()


# In[22]:


# missing_4 = missing_values(df_4)
# missing_4


# In[22]:





# In[22]:





# In[23]:


# Lucky to find that the first utterance is a problem 
# BASE_CORPUS Version
# [df.head(1)][0]["text"][0]


# In[24]:


# [df_1.head(1)][0]["text"][0]


# In[25]:


# [df_2.head(1)][0]["text"][0]


# In[26]:


# clean([df_2.head(1)][0]["text"][0])

# Returns None
# [df_3.head(1)][0]["text"][0] 


# In[27]:


# assert df_4["text"].all() == df_2["text"].all() == df_1["text"].all() 
# Damn why God Why did I waste time then? 


# In[28]:


# Option 5 - Use lambda Function
df["text_clean"] = df["text"].apply(lambda row : clean(row))


# In[29]:


# assert df_4["text"][0] == df["text_clean"][0] 
# Yes it should not be truee so wohhoo!!!


# In[30]:


df["text"][0]


# In[31]:


df["text_clean"][0]


# ### Prep for the Bert thingy

# In[32]:


# making a deep copy coz I think pandas works with shallow copies by default! 
train_df = df[["text_clean", "meta.majority_type", "meta.annotation-types"]].copy() 
train_df.rename(columns={"meta.majority_type" : "label", "meta.annotation-types" : "options"}, inplace=True)


# In[33]:


# no need of index
train_df.reset_index(drop=True, inplace=True)


# In[34]:


train_df.head()


# In[35]:


# Dealing with categorical data
codes, labels = pd.factorize(train_df["label"])


# In[36]:


train_df["label"] = pd.Series(data=codes).copy()


# In[37]:


train_df.head()


# In[39]:


train_df["label"].value_counts()
#-1 means that these are the labels that are missing


# In[40]:


train_df["num_tokens"] = train_df["text_clean"].apply(lambda sent : len(re.findall(r'\w+', sent)))


# In[41]:


# Removed all the token that exceeded the limit
train_df = train_df[train_df["num_tokens"]<510]


# In[42]:


len(train_df["num_tokens"])


# In[45]:


train_df["label"].value_counts()


# In[54]:


train_df_missing = train_df.loc[train_df["label"]<0]
train_df_missing["label"].value_counts()


# In[52]:


train_df_labeled = train_df.loc[train_df["label"]>=0]
train_df_labeled["label"].value_counts()


# In[55]:


# Checks if things are in order so far
assert len(train_df_labeled.loc[train_df_labeled["label"]<0]) == 0 
assert len(train_df)-len(train_df_missing) == len(train_df_labeled)


# In[56]:


train_df_labeled.head()


# In[57]:


train_df_labeled["label"].value_counts()


# In[58]:


train_df_labeled["num_tokens"].value_counts()


# In[59]:


test = train_df_labeled.iloc[3333]
print(test["text_clean"])
print(test["num_tokens"])
print(test["label"])
print(labels)


# In[64]:


train_df["num_tokens"].value_counts()


# In[60]:


train_df_missing["label"].value_counts()


# ### BERT variants
# All the BERTs that have been trained on conversational data can be more benifical that just the vanilla bert trained on something like Wiki corpus.
# 
# 1. [ ] MPC-BERT : Trained on Multi-Party comm
# 2. [ ] CS-BERT : Customer Service
# 3. [ ] RobertA - [Link](https://huggingface.co/roberta-large-mnli)

# 

# In[61]:


from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#Original:  Our friends won't buy this analysis, let alone the next one we propose.
#Tokenized:  ['our', 'friends', 'won', "'", 't', 'buy', 'this', 'analysis', ',', 'let', 'alone', 'the', 'next', 'one', 'we', 'propose', '.']
#Token IDs:  [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]


# In[65]:


max_len = 0
all_sentences = train_df["text_clean"]
# For every sentence...
for sent in all_sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)


# For the Hyper- for the sweep configurations we should follow : [Link](https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it)
# 
# The ones used in this block below are the ones that are suggested in the actual bert paper

# In[89]:


import wandb
sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {

        'learning_rate': {
            'values': [ 5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [4, 8]
        },
        'epochs':{
            'values':[2, 3, 4]
        }
    }
}
# sweep_defaults = {
#     'learning_rate': 5e-5,
#     'batch_size': 32,
#     'epochs':2
# }

sweep_id = wandb.sweep(sweep_config)


# #### Tokenization
# To follow a general convention that the sizes should be in powers of 2, we’ll choose the closest number that is a power of 2, i.e, 64.
# 
# Now, we’re ready to perform the real tokenization. But as we’re using transformers, we can use an inbuilt function tokenizer.encode_plus which automates all of the following tasks:
# 
# 1. Split the sentence into tokens.
# 2. Add the special `[CLS]` and `[SEP]` tokens.
# 3. Map the tokens to their IDs.
# 4. Pad or truncate all sentences to the same length.
# 5. Create the attention masks which explicitly differentiate real tokens from `[PAD]` tokens.
# 

# In[67]:


# Remember we are only working iwth the training set first as the label comparision for the missing labels 10% is still left - so for now this(train_df_labeled) would be ou entire dataset
# labeled_sentences = train_df_labeled["text_clean"]
# given_labels = train_df_labeled["label"]


# In[68]:


# labeled_sentences.iloc[35]


# In[69]:


# assert len(labeled_sentences) == len(given_labels)


# In[74]:


# import torch
# Tokenize all of the sentences and map the tokens to thier word IDs.
# input_ids = []
# attention_masks = []

# For every sentence...
for sent in labeled_sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(given_labels, dtype=torch.int)

output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')


# In[75]:


# Print sentence 0, now as a list of IDs.
print('Original: ', labeled_sentences[0])
print('Token IDs:', input_ids[0])


# #### Train - Test Split
# 
# 

# In[76]:


from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"{train_size} training samples present")
print(f"{val_size} validation samples present")


# In[78]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
# WANDB PARAMETER
def ret_dataloader():
    batch_size = wandb.config.batch_size
    print('batch_size = ', batch_size)
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader,validation_dataloader


# In[79]:


from transformers import BertForSequenceClassification, AdamW, BertConfig

def ret_model():

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels = 2, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    return model


# In[80]:


def ret_optim(model):
    print('Learning_rate = ',wandb.config.learning_rate )
    optimizer = AdamW(model.parameters(),
                      lr = wandb.config.learning_rate, 
                      eps = 1e-8 
                    )
    return optimizer


# In[81]:


from transformers import get_linear_schedule_with_warmup

def ret_scheduler(train_dataloader,optimizer):
    epochs = wandb.config.epochs
    print('epochs =>', epochs)
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    return scheduler


# In[83]:


import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# There's a ton to explore here with wandb man - [video](https://www.youtube.com/watch?v=9zrmUIlScdY) 
# 
# 1. [ ] Use various models in the yaml file?
# 2. [ ] Use different host machines - We could use lab computers to train/tune the hyperparameters

# #### Training Function

# In[85]:


import random
import numpy as np

def train():
  wandb.init()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  
  model = ret_model()
  model.to(device)
  train_dataloader, validation_dataloader = ret_dataloader()
  
  optimizer = ret_optim(model)
  
  scheduler = ret_scheduler(train_dataloader, optimizer)
  
  training_stats = []
  total_t0 = time.time()
  epochs = wandb.config.epochs

  for epoch_i in range(0,epochs):
    
    #Training
    print(f'========== EPOCH {epoch_i+1} / {epochs} =========')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
      if step % 40 == 0 and not step == 0:
        elapsed = format_time(time.time()-t0)

        print(f" Batch {step} of {len(train_dataloader)}.     Elapsed : {elapsed}")
      
      # UNpackign the batch data and sending to gpu
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)

      model.zero_grad()

      loss, logits = model(b_input_ids, 
                            token_type_ids = None,
                            attention_mask = b_input_mask,
                            labels = b_labels)
      
      #Log the train loss in WandB
      wandb.log({'train_batch_loss': loss.item()})
      total_train_loss += loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
    

    avg_train_loss = total_train_loss/len(train_dataloader)
    training_time = format_time(time.time() - t0)

    #Log the avg train loss
    wandb.log({'avg_trin_loss' : avg_train_loss})
    print("")
    
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    
    # Validation
    print("Running Validation ...")
    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    #Evaluatea dat for each epoch 
    for batch in validation_dataloader:
      # UNpackign the batch data and sending to gpu
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)

      # No BP - Validation
      with torch.no_grad:
        (loss, logits) = model(b_input_ids, 
                            token_type_ids = None,
                            attention_mask = b_input_mask,
                            labels = b_labels)
        
      total_eval_loss += loss.item()

      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy/len(validation_dataloader)
    print("     Accuracy : {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    #Log Avg val accuracy 
    wandb.log({"val_accuracy" : avg_val_accuracy, 'avg_val_loss' : avg_val_loss})
    print("     Validation Loss : {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0))) 


# In[91]:


wandb.agent(sweep_id,function=train)


# In[82]:


# import random
# import numpy as np

#     # This training code is based on the `run_glue.py` script here:
#     # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

#     # Set the seed value all over the place to make this reproducible.
# def train():
#     wandb.init(config=sweep_defaults)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     model = ret_model()
#     model.to(device)
#     #wandb.init(config=sweep_defaults)
#     train_dataloader,validation_dataloader = ret_dataloader()
#     optimizer = ret_optim(model)
#     scheduler = ret_scheduler(train_dataloader,optimizer)

#     #print("config ",wandb.config.learning_rate, "\n",wandb.config)
#     seed_val = 42
   
#     random.seed(seed_val)
#     np.random.seed(seed_val)
#     torch.manual_seed(seed_val)
#     #torch.cuda.manual_seed_all(seed_val)

#     # We'll store a number of quantities such as training and validation loss, 
#     # validation accuracy, and timings.
#     training_stats = []

#     # Measure the total training time for the whole run.
#     total_t0 = time.time()
#     epochs = wandb.config.epochs
#     # For each epoch...
#     for epoch_i in range(0, epochs):
        
#         # ========================================
#         #               Training
#         # ========================================
        
#         # Perform one full pass over the training set.

#         print("")
#         print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#         print('Training...')

#         # Measure how long the training epoch takes.
#         t0 = time.time()

#         # Reset the total loss for this epoch.
#         total_train_loss = 0

#         # Put the model into training mode. Don't be mislead--the call to 
#         # `train` just changes the *mode*, it doesn't *perform* the training.
#         # `dropout` and `batchnorm` layers behave differently during training
#         # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
#         model.train()

#         # For each batch of training data...
#         for step, batch in enumerate(train_dataloader):

#             # Progress update every 40 batches.
#             if step % 40 == 0 and not step == 0:
#                 # Calculate elapsed time in minutes.
#                 elapsed = format_time(time.time() - t0)
                
#                 # Report progress.
#                 print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

#             # Unpack this training batch from our dataloader. 
#             #
#             # As we unpack the batch, we'll also copy each tensor to the GPU using the 
#             # `to` method.
#             #
#             # `batch` contains three pytorch tensors:
#             #   [0]: input ids 
#             #   [1]: attention masks
#             #   [2]: labels 
#             b_input_ids = batch[0].to(device)
#             b_input_mask = batch[1].to(device)
#             b_labels = batch[2].to(device)

#             # Always clear any previously calculated gradients before performing a
#             # backward pass. PyTorch doesn't do this automatically because 
#             # accumulating the gradients is "convenient while training RNNs". 
#             # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
#             model.zero_grad()        

#             # Perform a forward pass (evaluate the model on this training batch).
#             # The documentation for this `model` function is here: 
#             # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
#             # It returns different numbers of parameters depending on what arguments
#             # arge given and what flags are set. For our useage here, it returns
#             # the loss (because we provided labels) and the "logits"--the model
#             # outputs prior to activation.
#             outputs = model(b_input_ids, 
#                                 token_type_ids=None, 
#                                 attention_mask=b_input_mask, 
#                                 labels=b_labels)
#             loss, logits = outputs['loss'], outputs['logits']
#             wandb.log({'train_batch_loss':loss.item()})
#             # Accumulate the training loss over all of the batches so that we can
#             # calculate the average loss at the end. `loss` is a Tensor containing a
#             # single value; the `.item()` function just returns the Python value 
#             # from the tensor.
#             total_train_loss += loss.item()

#             # Perform a backward pass to calculate the gradients.
#             loss.backward()

#             # Clip the norm of the gradients to 1.0.
#             # This is to help prevent the "exploding gradients" problem.
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             # Update parameters and take a step using the computed gradient.
#             # The optimizer dictates the "update rule"--how the parameters are
#             # modified based on their gradients, the learning rate, etc.
#             optimizer.step()

#             # Update the learning rate.
#             scheduler.step()

#         # Calculate the average loss over all of the batches.
#         avg_train_loss = total_train_loss / len(train_dataloader)            
        
#         # Measure how long this epoch took.
#         training_time = format_time(time.time() - t0)

#         wandb.log({'avg_train_loss':avg_train_loss})

#         print("")
#         print("  Average training loss: {0:.2f}".format(avg_train_loss))
#         print("  Training epcoh took: {:}".format(training_time))
            
#         # ========================================
#         #               Validation
#         # ========================================
#         # After the completion of each training epoch, measure our performance on
#         # our validation set.

#         print("")
#         print("Running Validation...")

#         t0 = time.time()

#         # Put the model in evaluation mode--the dropout layers behave differently
#         # during evaluation.
#         model.eval()

#         # Tracking variables 
#         total_eval_accuracy = 0
#         total_eval_loss = 0
#         nb_eval_steps = 0

#         # Evaluate data for one epoch
#         for batch in validation_dataloader:
            
#             # Unpack this training batch from our dataloader. 
#             #
#             # As we unpack the batch, we'll also copy each tensor to the GPU using 
#             # the `to` method.
#             #
#             # `batch` contains three pytorch tensors:
#             #   [0]: input ids 
#             #   [1]: attention masks
#             #   [2]: labels 
#             b_input_ids = batch[0].cuda()
#             b_input_mask = batch[1].to(device)
#             b_labels = batch[2].to(device)
            
#             # Tell pytorch not to bother with constructing the compute graph during
#             # the forward pass, since this is only needed for backprop (training).
#             with torch.no_grad():        

#                 # Forward pass, calculate logit predictions.
#                 # token_type_ids is the same as the "segment ids", which 
#                 # differentiates sentence 1 and 2 in 2-sentence tasks.
#                 # The documentation for this `model` function is here: 
#                 # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
#                 # Get the "logits" output by the model. The "logits" are the output
#                 # values prior to applying an activation function like the softmax.
#                 outputs = model(b_input_ids, 
#                                       token_type_ids=None, 
#                                       attention_mask=b_input_mask,
#                                       labels=b_labels)
#                 loss, logits = outputs['loss'], outputs['logits']
                
#             # Accumulate the validation loss.
#             total_eval_loss += loss.item()

#             # Move logits and labels to CPU
#             logits = logits.detach().cpu().numpy()
#             label_ids = b_labels.to('cpu').numpy()

#             # Calculate the accuracy for this batch of test sentences, and
#             # accumulate it over all batches.
#             total_eval_accuracy += flat_accuracy(logits, label_ids)
            

#         # Report the final accuracy for this validation run.
#         avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#         print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

#         # Calculate the average loss over all of the batches.
#         avg_val_loss = total_eval_loss / len(validation_dataloader)
        
#         # Measure how long the validation run took.
#         validation_time = format_time(time.time() - t0)
#         wandb.log({'val_accuracy':avg_val_accuracy,'avg_val_loss':avg_val_loss})
#         print("  Validation Loss: {0:.2f}".format(avg_val_loss))
#         print("  Validation took: {:}".format(validation_time))

#         # Record all statistics from this epoch.
#         training_stats.append(
#             {
#                 'epoch': epoch_i + 1,
#                 'Training Loss': avg_train_loss,
#                 'Valid. Loss': avg_val_loss,
#                 'Valid. Accur.': avg_val_accuracy,
#                 'Training Time': training_time,
#                 'Validation Time': validation_time
#             }
#         )

#     print("")
#     print("Training complete!")

#     print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

