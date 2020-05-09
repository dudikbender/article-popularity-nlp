from wordcloud import WordCloud
import matplotlib.pyplot as plt

def Generate_Wordcloud(articles, background='white', max_words=500, figsize=(12,6)):
    '''Generates a wordcloud image from a specified dataset of articles'''
    # Join the different processed titles of the articles together.
    title_string = ','.join(list(articles.title.values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color=background, max_words=max_words, contour_width=3, contour_color='grey')
    # Generate a word cloud
    wordcloud.generate(title_string)
    # Visualize the word cloud
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()