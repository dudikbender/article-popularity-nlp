import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


def dummy_regressor(X_train, X_test, y_train, y_test):
    dummy_clf = DummyRegressor()
    dummy_clf.fit(X_train, y_train)
    dummy_pred = dummy_clf.predict(X_test)

    # Evaluate the squared MSE
    calculated_error = np.sqrt(mean_squared_error(y_test, dummy_pred))
    return round(calculated_error,3)

def vectorizer_predictor(X_train, X_test, y_train, y_test, vectorizer, model):
    '''Transforms the text data set using specified vectorizer, then predicts continuous value using RandomForestRegressor'''
    vect = vectorizer(stop_words='english')
    count_X_train = vect.fit_transform(X_train)
    # Prepare testing data
    count_X_test = vect.transform(X_test)
    # Instantiate the model
    model = model
    # Fit the model using the training array and training labels
    model.fit(count_X_train, y_train)
    # Run model on the testing data
    y_pred = model.predict(count_X_test)
    # Evaluate using squared mean squared error
    calculated_error = np.sqrt(mean_squared_error(y_test, y_pred))
    return round(calculated_error,3)

class ArticleModeller():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.test_size = 0.2
        self.seed = 1
    
    def run_dummy_regressor(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=self.test_size, random_state=self.seed)
        result = dummy_regressor(X_train, X_test, y_train, y_test)
        print('Dummy Regressor Mean Squared Error is: {}'.format(result))

    def vectorizer_prediction(self, vectorizer, model):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=self.test_size, random_state=self.seed)
        result = vectorizer_predictor(X_train, X_test, y_train, y_test, vectorizer, model)
        print('{} Mean Squared Error is: {}'.format(vectorizer, result))
    
    def run_dummy_and_vectorizers(self, vectorizers, vectorizer_names, model):
        '''Produces a comparison chart of predicted continuous values using specified vectorizers and model,
        compared against dummy vectorizer and dummy model'''
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=self.test_size, random_state=self.seed)
        dummy_result = dummy_regressor(X_train, X_test, y_train, y_test)

        methods = ['Dummy Vectorizer']
        results = [dummy_result]

        # Then iterate through the vectorizer options
        for i in range(len(vectorizers)):
            method = vectorizer_names[i]
            result = vectorizer_predictor(X_train, X_test, y_train, y_test, vectorizers[i], model)

        # Then add the vectorizer method and results
            methods.append(method)
            results.append(result)
        
        df = pd.DataFrame({'methods':methods,
                          'results':results})

        # Visualise the dataframe of methods and results
        plt.figure(figsize=(10, 7.5))
        g = sns.barplot('methods', 'results', data=df)
        for index, row in df.iterrows():
            plt.text(index, row.results + (row.results * 0.01), row.results, color='black', ha='center', size=12)

        plt.title('Comparison of Prediction Errors, Dummy Vectorizer to Actual Vectorizers', fontsize=16)
        g.set_xlabel("Methods",fontsize=12)
        g.set_ylabel("Mean Squared Error",fontsize=12)
        g.tick_params(labelsize=14)
        plt.show()


import spacy
import en_core_web_sm
from sklearn.manifold import TSNE
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure

from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
from IPython.core.display import display, HTML

# Initialise a Count Vectorizer for the entire article text corpus
def fit_lda(text,  number_of_topics):

    LDA_vect = CountVectorizer(stop_words='english') 
    LDA_count_data = LDA_vect.fit_transform(text)

    # Create and fit the LDA model
    topic_count = number_of_topics
    lda = LDA(n_components=topic_count, n_jobs=-1)
    lda.fit(LDA_count_data)

output_notebook() # this will ensure that the Bokeh plot will show in the notebook (import output_file to create separate file)
def tsne_plot(text, vectors, plot_width=500, plot_height=500, title='INSERT TITLE HERE'):
    '''Creates a plot representing the 2-dimensional t-SNE reduction of the article vectors'''

    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', random_state=46)
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    # Create a data source for labelling the points in the plot
    data_source = {'title':text,'x':x,'y':y}
    source = ColumnDataSource(data_source)
    
    # This will create the hover functionality, for better readability
    hover = HoverTool(tooltips=[("Article Title", '@title')])
    
    p = figure(plot_width=plot_width, plot_height=plot_height, tools=['box_zoom','reset',hover],
           title=title, toolbar_location="below")

    p.circle('x','y', size=10, source=source)
    p.axis.visible = False

    show(p)


class TopicModeller():

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.topic_count = 5

    def generate_word_vectors(self, column):
        nlp = spacy.load('en_core_web_sm')
        docs = [ nlp(text) for text in self.dataframe[column] ]
        vectors = [ word.vector for word in docs]
        text = [ word.text for word in docs]

        tsne_plot(text, vectors, title='Article Titles Grouped by Word Vectors')

    def generate_lda_visualisation(self, text):
        LDA_vect = CountVectorizer(stop_words='english') 
        LDA_count_data = LDA_vect.fit_transform(text)

        # Create and fit the LDA model
        lda = LDA(n_components=self.topic_count, n_jobs=-1)
        lda.fit(LDA_count_data)

        # Generate the LDA visualisation
        LDAvis_prepared = sklearn_lda.prepare(lda, LDA_count_data, LDA_vect)
        path = './ldavis_prepared_'+ str(self.topic_count) +'.html'
        pyLDAvis.save_html(LDAvis_prepared, path)
        display(HTML(path))