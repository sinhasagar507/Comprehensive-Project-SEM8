import re
import emoji, emot

# from afinn import Afinn

# Instantiate afinn
# aff = Afinn()

# Initialize emot object
emot_object = emot.core.emot()


def clean(text, newline=True, quote=True, bullet_point=True, dates=True,
          link=True, strikethrough=True, spoiler=True, heading=True, emoji=True, emoticon=True, contraction=True):
    text = str(text)

    # Newlines we don't need - only
    if newline:
        text = re.sub(r'\n+', ' ', text)
        # Remove the many " " that we replaced in the last step
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
        # Implement the emoji scheme here
        # Implementing a Naive Emoji Scheme
        # Some associated libraries are EMOT and DEMOJI
        text = emoji.demojize(text).replace(":", "").replace("_", "")
        # Makes more sense for the node feature but might as well import that function here if ready

    if dates:
        text = re.sub(r'(\d+/\d+/\d+)', '', text)

    if emoticon:
        # Implement the emoticon scheme here.
        # Makes more sense for the node feature but might as well import that function here if ready
        pass

    # Needs to be the last step in the process
    # if contractions:
    # text = contractions.fix(text)
    # print("Running")
    return text


# Give a description of metadata in the IPython Notebook
# Requirements
# Consider only English Tweets
# Remove all URLs, @tags, whitespaces, newlines, whitespaces and whitespace characters, bracketed words, special characters, sentence within double quotes
# Resolve the character encoding issue - â€˜
# Replace , and ; by 'and'
# Take care of HTML encodings - &amp - means &
# Replace full stops in submissions and comments by an <EOS> token
# Encode hashtags - umm..., how they can be encoded? Will have to study them closely. Save them in a separate column. They can be used as metadata to the nodes
# If a double quote is found - only include the sentence within quotes
# Expand all contractions
# Remove full stops that aren't ellipsis


def clean_twitter(text):
    pass


# Resolve Afinn Later
# def afinn_sentiment_score(utterances):
#     # Compute polarity scores and assign labels
#     scores = [aff.score(utterance) for utterance in utterances]
#     mean_score = mean(scores)
#     return mean_score


def count_emojis(utterance):
    # pattern = "^[0-9A-F]{3, }"
    # return len(re.findall(pattern, utterance))
    emot_dict = emot_object.emoji(utterance)
    return len(emot_dict['value'])

# OOP's Standard
# class Utilities:
#
#     def __init__(self, text):
#         self.text = text
#
#     def clean_again(self, newline=True, quote=True, bullet_point=True, dates=True, link=True, strikethrough=True,
#                     spoiler=True, heading=True, emoji=True, emoticon=True, contraction=True):
#
#         self.text = str(self.text)
#
#         # Newlines we don't need - only
#
#         if newline:
#             self.text = re.sub(r'\n+', ' ', self.text)
#
#             # Remove the many " " that we replaced in the last step
#             self.text = self.text.strip()
#             self.text = re.sub(r'\s\s+', ' ', self.text)
#
#         # > are for the quoted texts from the main comment or the reply
#         if quote:
#             self.text = re.sub(r'>', '', self.text)
#
#         # Bullet points/asterisk are used for markdown like - bold/italic - Could create trouble in parsing? idk
#         if bullet_point:
#             self.text = re.sub(r'\*', '', self.text)
#             self.text = re.sub('&amp;#x200B;', '', self.text)
#
#         # []() Link format then we remove both the tag/placeholder and the link
#         if link:
#             self.text = re.sub(r"http\S+", '', self.text)
#             self.text = re.sub(r'\[.*?\]\(.*?\)', '', self.text)
#
#         # Strikethrough
#         if strikethrough:
#             self.text = re.sub('~', '', self.text)
#
#         # Spoiler, which is used with < less-than (Preserves the text)
#         if spoiler:
#             self.text = re.sub('&lt;', '', self.text)
#             self.text = re.sub(r'!(.*?)!', r'\1', self.text)
#
#         # Heading to be removed as there are these markdown style features in reddit too
#         if heading:
#             self.text = re.sub('#', '', self.text)
#
#         if emoji:
#             # Implement the emoji scheme here.
#             # Makes more sense for the node feature but might as well import that function here if ready
#             pass
#
#         if dates:
#             self.text = re.sub(r'(\d+/\d+/\d+)', '', self.text)
#
#         if emoticon:
#             # Implement the emoticon scheme here.
#             # Makes more sense for the node feature but might as well import that function here if ready
#             pass
#
#         # Needs to be the last step in the process
#         # if contractions:
#         # text = contractions.fix(text)
#         # print("Running")
#         return self.text


# count_emojis("This is thumbs up: 👍, and this is thumbs up with dark skin tone: 👍🏿")
# Example sentences for test:
# "Why Bihar is the nucleus of arson &amp; loot for Agnipath Scheme?\n\nAfter some thought, I guess, I've found the answer. Tejaswi Yadav visited with RaGa to address â€˜Ideas of Indiaâ€™ event in London
#  \n\nThe toolkit was hatched there,funds arranged &amp; RJD goons are creating ruckus. Your take?"
