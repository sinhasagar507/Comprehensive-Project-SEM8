import re
import emot
import contractions as cm

# from afinn import Afinn

# Instantiate afinn
# aff = Afinn()

# Initialize emot object
emot_object = emot.core.emot()


def clean(text, newline=True, quote=True, bullet_point=True, dates=True,
          link=True, strikethrough=True, spoiler=True, heading=True, emoji=True, emoticon=True, condensed=True):
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


# Requirements
# Done List
# URLs and tags
# Remove one or more occurrence of spaces with " " - a single space


# List of Special Characters to be treated: [~, ', !, @, #, $, %, ^, &, *, (, ), -, _, +, =, {, }, [, ], |, \, /, :, ;, ", ', <, >, ,, ., ?]
# List of Special Characters to be removed: [%, ^, *, -, _, +, =, |, \, /, !]
# Replace # with and. Treat hashtags separately
# List of special characters that remain to be treated: [:, ;, ,, ., ?, !]

def clean_twitter(text_twitter, urls=True, tags=True, newLine=True, ellipsis=True,
                  ampersand=True, tilde=True, special_chars=True, dollar=True, commas_semicols=True,
                  bracketed_phrases=True, contractions=True, quotation_marks=True, greater_than_less_than=True,
                  question_mark_exclaim=True, character_encodings=True, trademark=True, condensed=True):
    if urls:
        url_pattern = "https?:\/\/(www\.)?(\w+)(\.\w+)\/\w*"
        text_twitter = re.sub(url_pattern, "", text_twitter)

    if tags:
        text_twitter = re.sub("@\w+", "", text_twitter)

    # Remove "\n". One or more occurrences
    if newLine:
        # Replacing single occurrences of '\n' with ''
        # Replacing multiple occurrences, i.e., >=2 occurrences with '.'
        text_twitter = re.sub("\n", "", text_twitter)
        text_twitter = re.sub("\n\n", ".", text_twitter)
        text_twitter = text_twitter.strip()

    # Fix contractions
    if condensed:
        text_twitter = cm.fix(text_twitter)
        text_twitter = re.sub("\s\s", "", text_twitter)

    # Remove "ellipsis"
    if ellipsis:
        text_twitter = re.sub("\.{2,}", "", text_twitter)

    # Replace "&" with "and"
    if ampersand:
        text_twitter = re.sub("&", "and", text_twitter)

    # Replace "~" with "about"
    if tilde:
        text_twitter = re.sub("~", "about", text_twitter)

    # Remove the special_chars list: [%, ^, *, -, _, +, =, |, \, /, ?]
    if special_chars:
        spec_char_list = ['%', '^', '*', '-', '_', '+', '=', '|', '/', '?']
        sent = ""
        new_sent_tokens = []

        for character in text_twitter:
            if str(character) not in spec_char_list:
                new_sent_tokens.append(character)

        sent = sent.join(new_sent_tokens)
        sent = sent.strip()
        text_twitter = sent

    # Rename $ as dollar
    # if dollar:
    #     text_twitter = re.sub("$", "dollar", text_twitter)

    # Remove brackets and any text enclosed within simple brackets, usually used for acronyms
    if bracketed_phrases:
        text_twitter = re.sub("\(\w+\)", "", text_twitter)

    # If single quotes or double quotes have been used in tweets, encash their meaning for the time being. Don't include any other information
    if quotation_marks:
        text_twitter = re.sub("(\'|\")[a-zA-Z0-9\s+\.]*(\'|\")", "", text_twitter)

    # For the time being, replace commas by "" and semicolons by "."
    if commas_semicols:
        text_twitter = re.sub("\,+", "", text_twitter)
        text_twitter = re.sub("\;+", ".", text_twitter)

    # Resolve '>' and '<'
    # Replace these characters with their respective names
    if greater_than_less_than:
        text_twitter = re.sub("<", "is less than", text_twitter)
        text_twitter = re.sub(">", "is greater than", text_twitter)
        text_twitter = re.sub("<=", "is less than or equal to", text_twitter)
        text_twitter = re.sub(">=", "is greater than or equal to", text_twitter)

    # For the time being, replace and interjections with a full stop
    if question_mark_exclaim:
        text_twitter = re.sub("(\?|\!)+", ".", text_twitter)

    # Resolve character encodings
    if character_encodings:
        text_twitter = re.sub("â|€|¦|â|€˜|€™", "", text_twitter)

    # Remove trademark symbol
    if trademark:
        text_twitter = re.sub("\u2122", "", text_twitter)

    return text_twitter


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


# Observations
#  Raw data was extracted from Twitter. Characters in Hindi and other regional languages had encoding issues. We removed them
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
