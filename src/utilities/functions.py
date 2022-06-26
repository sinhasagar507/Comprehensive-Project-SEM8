"""
General Class Utility Functions
"""

# Import standard libraries
import re

# Import third-party libraries
import emot
import emoji
import contractions as cm

import nltk
from nltk.stem import WordNetLemmatizer

# Downloading the relevant libraries and dependencies in NLTK module for preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

emot_object = emot.core.emot()  # Initialize Emoji Object
lemmatizer = WordNetLemmatizer()  # Initialize the NLTK Lemmatizer


def extract_hashtags(utterance):
    """ Returns all Twitter hashtags from a tweet

    Store all hashtags from a tweet and store them in a separate column
    """
    hashtags_ls = re.findall("#\w+", utterance)
    return hashtags_ls


def extract_username_tags(utterance):
    """ Returns all Username tags from a tweet

    Store all tags from a tweet and store them in a separate column
    """

    username_tags = re.findall("@\w+", utterance)
    return username_tags


def clean_reddit(
        text_reddit, newline=True, quote=True,
        bullet_point=True, dates=True, link=True,
        strikethrough=True, spoiler=True, heading=True,
        emoj=True, emoticon=True, condensed=True):
    # Newlines we don't need - only
    if newline:
        text_reddit = re.sub(r'\n+', ' ', text_reddit)
        # Remove the many " " that we replaced in the last step
        text_reddit = text_reddit.strip()
        text_reddit = re.sub(r'\s\s+', ' ', text_reddit)

    # > are for the quoted texts from the main comment or the reply
    if quote:
        text_reddit = re.sub(r'>', '', text_reddit)

    # Bullet points/asterisk are used for markdown like - bold/italic - Could create trouble in parsing? idk
    if bullet_point:
        text_reddit = re.sub(r'\*', '', text_reddit)
        text_reddit = re.sub('&amp;#x200B;', '', text_reddit)

    # []() Link format then we remove both the tag/placeholder and the link
    if link:
        text_reddit = re.sub(r"http\S+", '', text_reddit)
        text_reddit = re.sub(r'\[.*?\]\(.*?\)', '', text_reddit)

    # Strikethrough
    if strikethrough:
        text_reddit = re.sub('~', '', text_reddit)

    # Spoiler, which is used with < less-than (Preserves the text)
    if spoiler:
        text_reddit = re.sub('&lt;', '', text_reddit)
        text_reddit = re.sub(r'!(.*?)!', r'\1', text_reddit)

    # Heading to be removed as there are these markdown style features in reddit too
    if heading:
        text_reddit = re.sub('#', '', text_reddit)

    if emoj:
        # Implement the emoji scheme here
        # Implementing a Naive Emoji Scheme
        # Some associated libraries are EMOT and DEMOJI
        # text_reddit = emoji.demojize(text_reddit).replace(":", "").replace("_", "")
        # Makes more sense for the node feature but might as well import that function here if ready
        pass

    if dates:
        text_reddit = re.sub(r'(\d+/\d+/\d+)', '', text_reddit)

    if emoticon:
        # Implement the emoticon scheme here.
        # Makes more sense for the node feature but might as well import that function here if ready
        pass

    # Needs to be the last step in the process
    # if contractions:
    # text = contractions.fix(text)
    # print("Running")
    return text_reddit


def clean_twitter(
        text_twitter, urls=True, tags=True,
        newLine=True, ellipsis=True, ampersand=True,
        tilde=True, special_chars=True, dollar=True,
        commas_semicols=True, bracketed_phrases=True, contractions=True,
        quotation_marks=True, greater_than_less_than=True, question_mark_exclaim=True,
        character_encodings=True, trademark=True, condensed=True):
    """ The Twitter Clean Methodology
    Clean tweets after extracting all hashtags and username tags
    Not comprehensive enough to capture all idiosyncrasies, but works most of the time
    """
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
        text_twitter = text_twitter.replace("&amp", "")

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


def convert_to_lower(text):
    """ This function block performs twitter text normalization
        Hate, HATE, haTE, etc.
    """

    exclude_tags_list = ['NN', 'NNS', 'NNP', 'NNPS']  # Check if the attached POS tags are correct or not
    modified_text_ls = []

    words = nltk.word_tokenize(text)  # Tokenize the sentence and extract POS tags

    words = [lemmatizer.lemmatize(word) for word in words]  # Perform lemmatization if required
    word_pos_tags = nltk.pos_tag(words)

    for (word, tag) in word_pos_tags:
        if tag not in exclude_tags_list:
            word = word.lower()
        modified_text_ls.append(word)

    text = ' '.join(modified_text_ls)

    return text


def count_emojis(utterance):
    """ Counts the total number of emojis in an utterance

    Can act a possible indicator of deception
    """
    emot_dict = emot_object.emoji(utterance)

    return len(emot_dict['value'])


def cnt_modifiers(text):
    """Count modifiers, i.e., adjectives and adverbs in an utterance
    The function block can detect probable deceptive clues in tweets and reddit posts
    """
    adj_pos_tags = ['JJ', 'JJR', 'JJS']  # POS tags describing adjectives
    adv_pos_tags = ['RB', 'RBR, RBS']  # POS tags for adverbs
    words = nltk.word_tokenize(text)
    word_tag_lst = nltk.pos_tag(words)
    cnt_tags = 0
    for (word, tag) in word_tag_lst:
        if tag in adj_pos_tags or tag in adv_pos_tags:
            cnt_tags += 1

    return cnt_tags


def pos_modal_vbs(text):
    """ Count the list of all modal verbs that indicate possibility, but not certainty
    The function block can detect probable deceptive clues in tweets and reddit posts
    """
    cnt_mods = 0
    pos_modal_ls = ['shall', 'should', 'can', 'could', 'will', 'would', 'may', 'must',
                    'might']  # List of 9 modal verbs indicating possibility

    words = text.split(" ")
    for word in words:
        if word in pos_modal_vbs:
            cnt_mods += 1
    return cnt_mods


# Count list of self-references
def cnt_self_ref(text):
    cnt_self = 0
    words = text.split()
    self_ref = ['I', 'me', 'mine', 'we', 'our', 'ours', 'us']  # Self-referencing pronouns

    for word in words:
        if word in self_ref:
            cnt_self += 1
    return cnt_self
