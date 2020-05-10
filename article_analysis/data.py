import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re

class ArticleDataset():
    def __init__(self, name):
        self.name = name

    def generate_wordcloud(self, column, background='white', max_words=500, size=(12,6)):
        '''Generates word cloud from text contain in specific column'''
        # Join the different processed titles of the articles together.
        title_string = ','.join(list(column.values))
        # Create a WordCloud object
        wordcloud = WordCloud(background_color=background, max_words=max_words, contour_width=3, contour_color='grey')
        # Generate a word cloud
        wordcloud.generate(title_string)
        # Visualize the word cloud
        plt.figure(figsize=size)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def exclude_outliers(self, data, sigma=2, negatives=False, rounded=0):
        '''Use this function to define the target variable counts that 
            you would like to exclude from your dataset. The default will exclude those
            indicate those row values that are greater than 2 standard deviations from the mean. Function returns 
            the cut-off values of selected sigma at upper and lower bounds and the
            n_top and n_bottom values to use to select a slice of your data.
    
            Variables:
            sigma - use to define the number of standard deviations from mean to return
        
            negatives - define whether the function should return negative lower bound values. Choose False for
                    one-tailed distributions.
        
            rounded - number of decimal places for the returned values'''
    
        mean = np.mean(data)
        std_dev = np.std(data)
        sigma_upper = mean + sigma * std_dev
        sigma_lower = mean - sigma * std_dev
        if negatives == True:
            sigma_lower = sigma_lower
        else:
            if sigma_lower < 0:
                sigma_lower = 0
    
        n_top = len(data[data > sigma_upper])
        n_bottom= len(data[data < sigma_lower])
    
        return round(sigma_upper, rounded), round(sigma_lower,rounded), n_top, n_bottom

    def make_paretto_chart(self,
                          dataframe,
                          column_name,
                          cutoff,
                          n_top,
                          title='Popularity of Article, Along with Contribution to Total',
                          x_axis_label='Sorted List',
                          y1_axis_label='Articles, Sorted by Popularity',
                          y2_axis_label='Cumulative Percentage',
                          include_color='darkblue',
                          exclude_color='red',
                          cutoff_line=True):
        ## Then plot the results in a Paretto chart, which indicates both the count and the cumulative percentage
        sorted_table = dataframe.sort_values(column_name, ascending=False).drop_duplicates().reset_index(drop=True) # Will also eliminate duplicates
        sorted_table['cumulative'] = sorted_table[column_name].cumsum()/sorted_table[column_name].sum()*100
        # Establish variables that will create the graph
        color_included = include_color
        color_excluded = exclude_color
        cutoff = cutoff
        bar_color = [color_included if x < cutoff else color_excluded for x in sorted_table.claps]

        ## Then plot the results to see what the ideal cut-off should be, indicated by a horizontal line
        fig, ax = plt.subplots(figsize=(12,8))
        ax.bar(sorted_table.index, sorted_table[column_name], color=bar_color, linewidth=0)
        ax.bar(sorted_table.index, sorted_table[column_name], color=bar_color, linewidth=0)
        plt.xticks(rotation=90, fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(sorted_table.index, [(x/100) for x in sorted_table['cumulative']], color="orange", marker="o", ms=1)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y1_axis_label)
        ax2.set_ylabel(y2_axis_label)

        ax.tick_params(axis="y", colors="grey")
        ax2.tick_params(axis="y", colors="grey")
        plt.title(title)
        if cutoff_line:
            ax.axhline(y=cutoff, ls='--', alpha=0.8)
        plt.show()

        print('The articles to exclude are:\n')
        for x in range(len(sorted_table[:n_top])):
            print('{} - {}: {} claps.'.format(x+1,sorted_table.iloc[x]['title'],sorted_table.iloc[x]['claps']))

import spacy
import en_core_web_sm
import nltk
from langdetect import detect
import string
import re

## Preprocess text for analysis

def remove_punct(text):
    text = re.sub(r'[^\w\s]','',text)
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def lower_all_text(text):
    text = text.lower()
    return text

def remove_stop_words(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = [ w for w in nltk.wordpunct_tokenize(text) if not w in stop_words ]
    text = str(" ".join(w for w in text))
    return text

def lemmatize_text(text):
    nlp = spacy.load('en_core_web_sm')
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

class TextProcessing():

    def __init__(self):
        self.data = []

    ## Combine these functions all together into one single preprocessing function
    def preprocess_text(self, 
                        detect_language=True,
                        remove_punctuation=True,
                        remove_special=True,
                        lower_all=True, 
                        stop_words=True,
                        lemmatize=True,
                        no_digits=False):
        '''Turn options on or off to modify the text changes that will be applied.
        
        Function returns a tuple of the modified text and the language detected (if that option is selected'''
        text = self

        if detect_language:
            language = detect(text)
        if remove_punctuation:
            text = remove_punct(text)
        if remove_special:
            text = remove_special_characters(text, remove_digits=no_digits)
        if lower_all==True:
            text = lower_all_text(text)
        if stop_words==True:
            text = remove_stop_words(text)
        if lemmatize:
            text = lemmatize_text(text)
        
        if detect_language:
            return text, language
        else:
            return text
    
    def list_of_entities(self):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self)
        article_entities = []
        for entity in doc.ents:
            if (entity.label_ == 'ORG') & (entity.text not in article_entities):
                article_entities.append(entity.text)
        
        return article_entities