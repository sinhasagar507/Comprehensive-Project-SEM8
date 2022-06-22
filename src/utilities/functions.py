import re
import emot

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
# Remove all URLs, @tags, newlines, whitespace characters, bracketed words, special characters
# If a sentence is within double quotes, only keep the context within the quotes. Remove other context for the time being
# Resolve the character encoding issue - â€˜
# Replace , and ; by 'and'
# Take care of HTML encodings - &amp - means &
# Replace full stops in submissions and comments by an <EOS> token
# Encode hashtags - umm..., how they can be encoded? Will have to study them closely. Save them in a separate column. They can be used as metadata to the nodes
# If a double quote is found - only include the sentence within quotes
# Convert all non-PROPER NOUN characters to lowercase
# Expand all contractions
# Remove full stops that aren't ellipsis


# First letter of every sentence - Capital Letter
# During tokenization, if the word is a proper noun, its first character should be in capital letters


# Done List
# URLs and tags
# Remove one or more occurrence of spaces with " " - a single space


# List of Special Characters to be treated: [~, ', !, @, #, $, %, ^, &, *, (, ), -, _, +, =, {, }, [, ], |, \, /, :, ;, ", ', <, >, ,, ., ?]
# List of Special Characters to be removed: [%, ^, *, -, _, +, =, |, \, /, !]
# Replace # with and. Treat hashtags separately
# List of special characters that remain to be treated: [:, ;, ,, ., ?, !]

def clean_twitter(text, urls=True, tags=True, newLine=True, ellipsis=True,
                  ampersand=True, tilde=True, special_chars=True, dollar=True, commas_semicols=True,
                  bracketed_phrases=True,
                  quotation_marks=True, greater_than_less_than=True, question_mark_exclaim=True,
                  character_encodings=True, trademark=True):
    if urls:
        text = re.sub("https?:\/\/(www\.)?(\w+)(\.\w+)\/\w*", "", text)

    if tags:
        text = re.sub("@\w+", "", text)

    # Remove any character between newlines

    # Remove "\n". One or more occurrences
    if newLine:
        text = re.sub("\n+", ".", text)
        text = text.strip()
    #
    # # Remove "ellipsis"
    if ellipsis:
        text = re.sub("\.{2,}", "", text)

    # Replace "&" with "and"
    if ampersand:
        text = re.sub("&", "and", text)

    # Replace "~" with "about"
    if tilde:
        text = re.sub("~", "about", text)

    # Remove the special_chars list: [%, ^, *, -, _, +, =, |, \, /, ?]
    if special_chars:
        spec_char_list = ['%', '^', '*', '-', '_', '+', '=', '|', '/', '?']
        sent = ""
        new_sent_tokens = []

        for character in text:
            if str(character) not in spec_char_list:
                new_sent_tokens.append(character)

        sent = sent.join(new_sent_tokens)
        sent = sent.strip()
        text = sent

    # Rename $ as dollar
    if dollar:
        text = re.sub("$", "dollar", text)

    # Remove brackets and any text enclosed within simple brackets, usually used for acronyms
    if bracketed_phrases:
        text = re.sub("\(\w+\)", "", text)

    # If single quotes or double quotes have been used in tweets, encash their meaning for the time being. Don't include any other information
    if quotation_marks:
        text = re.sub("(\'|\")[a-zA-Z0-9\s+\.]*(\'|\")", "", text)

    # For the time being, replace commas by "" and semicolons by "."
    if commas_semicols:
        text = re.sub("\,+", "", text)
        text = re.sub("\;+", ".", text)

    # Resolve '>' and '<'
    # Replace these characters with their respective names
    if greater_than_less_than:
        text = re.sub("<", "is less than", text)
        text = re.sub(">", "is greater than", text)
        text = re.sub("<=", "is less than or equal to", text)
        text = re.sub(">=", "is greater than or equal to", text)

    # Remove commas and interjections for the time being
    if question_mark_exclaim:
        text = re.sub("(\?|\!)+", "", text)

    # Resolve character encodings
    if character_encodings:
        text = re.sub("â|€|¦|â|€˜|€™", "", text)

    # Remove trademark symbol
    if trademark:
        text = re.sub("\u2122", "", text)

    return text


text = "Such beautiful Taquiyya! #NupurSharma should be â€˜forgivenâ€™ for what? For quoting from the scriptures? And the call for forgiveness comes NOW? AFTER the violence is done? â€˜Nupur Sharma should be forgiven as per Islam, says Jamaat Ulama-e-Hindâ€™!  https://t.co/U3s6DAjhFi"
print(clean_twitter(text))


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

print(clean_twitter("Those standing by Mohammed Zubair whose posts mocking Hindu Gods are currently viral, are the same folks who want Nupur Sharma jailed or dead.\n\nZubair should learn to respect other religions before making coded racial appeals to Islamists to run riot when someone debates Islam."))

pattern = "Such beautiful Taquiyya! #NupurSharma should be â€˜fo"
