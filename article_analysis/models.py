# General python and jupyter packages
import pandas as pd
import numpy as np
from IPython.core.display import display, HTML

# sci-kit learn packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Natural Language Processing packages
import spacy
import en_core_web_sm
from sklearn.manifold import TSNE
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
from tensorflow import keras

# Visualisation packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def dummy_regressor(X_train, X_test, y_train, y_test):
    dummy_clf = DummyRegressor()
    dummy_clf.fit(X_train, y_train)
    dummy_pred = dummy_clf.predict(X_test)

    # Evaluate the root mean squared error
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
    # Evaluate using root mean squared error
    calculated_error = np.sqrt(mean_squared_error(y_test, y_pred))
    return round(calculated_error,3)

def regression_model(X_train, X_test, y_train, y_test, model):
    # Instantiate the model
    model = model
    # Fit the model
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    # Return the root mean squared error
    rmse_error = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_error = mean_absolute_error(y_test, y_pred)
    return rmse_error, mae_error

def predict_from_input(input_text, X, y, model):
    # Apply the CountVectorizer
    vect = CountVectorizer(stop_words='english')
    X_train = vect.fit_transform(X) # for the training dataset of text
    X_test = vect.transform([input_text]) # for the new inputted text
    # Instantiate the model
    model = model
    # Fit the model
    model.fit(X_train, y)
    # Predict on test set
    predicted = model.predict(X_test)
    # Return the prediction
    return predicted[0]

class ArticleModeller():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.test_size = 0.2
        self.seed = 1

    def vectorizer_prediction(self, vectorizer, model):
        '''Returns root mean squared error using specific vectorizer and model.'''
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=self.test_size, random_state=self.seed)
        result = vectorizer_predictor(X_train, X_test, y_train, y_test, vectorizer, model)
        return result
    
    def run_dummy_and_vectorizers(self, vectorizers, vectorizer_names, model, figsize=(10,7.5),
                                 title='Comparison of Prediction Errors, Dummy Vectorizer to Actual Vectorizers'):
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
        plt.figure(figsize=figsize)
        g = sns.barplot('methods', 'results', data=df)
        for index, row in df.iterrows():
            plt.text(index, row.results + (row.results * 0.01), row.results, color='black', ha='center', size=12)

        plt.title(title, fontsize=16)
        g.set_xlabel("Methods",fontsize=12)
        g.set_ylabel("Root Mean Squared Error",fontsize=12)
        g.tick_params(labelsize=14)
        plt.show()

    def sklearn_prediction_with_vectors(self, model_select):
        ''' Will convert text to vectors using spaCy, then predict using specified regression model. Text data is passed from the object,
        so set the X and y inputs for the object.
        
        Inputs:
        self: the object with indicated X, y inputs

        model_select: regression model for use in prediction'''
        # Load spaCy's nlp package and convert words to vectors
        nlp = spacy.load('en_core_web_sm')
        vectors = [nlp(x).vector for x in self.X]
        X_train, X_test, y_train, y_test = train_test_split(vectors,self.y,test_size=self.test_size, random_state=self.seed)
        
        # Instantiate and train the model
        result = regression_model(X_train, X_test, y_train, y_test, model_select)
        return result

    def predict_popularity_from_input(self, input_text, model_select):
        X, y = self.X, self.y
        result = predict_from_input(input_text=input_text, X=X, y=y, model=model_select)
        return int(result)

# Bokeh packages
from bokeh.io import output_notebook, reset_output, show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure
reset_output()
output_notebook() # this will ensure that the Bokeh plot will show in the notebook (import output_file to create separate file)

# Initialise a Count Vectorizer for the entire article text corpus
def fit_lda(text,  number_of_topics):

    LDA_vect = CountVectorizer(stop_words='english') 
    LDA_count_data = LDA_vect.fit_transform(text)

    # Create and fit the LDA model
    topic_count = number_of_topics
    lda = LDA(n_components=topic_count, n_jobs=-1)
    lda.fit(LDA_count_data)

def tsne_plot(text, vectors, plot_width=500, plot_height=500, title='INSERT TITLE HERE'):
    '''Creates an interactive Bokeh plot representing the 2-dimensional t-SNE reduction of the article vectors,
    hover over the points to retrieve the associated text.'''
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
    # Create the figure, with the desired options
    p = figure(plot_width=plot_width, plot_height=plot_height, tools=['box_zoom','reset',hover],
           title=title, toolbar_location="below")
    p.circle('x','y', size=10, source=source)
    p.axis.visible = False
    show(p)


class TopicModeller():

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.topic_count = 5

    def generate_word_vector_plot(self, column):
        nlp = spacy.load('en_core_web_sm')
        docs = [ nlp(text) for text in self.dataframe[column] ]
        vectors = [ word.vector for word in docs]
        text = [ word.text for word in docs]
        # This will create the interactive TSNE plot for the specified text
        tsne_plot(text, vectors, title='Article Titles Grouped by Word Vectors')

    def generate_lda_visualisation(self, text):
        '''This function will create an interactive graphic using the text provided and topic count of the object.
        Also saves an html file in the local directory.

        self = dataframe object
        text = text to analyse and group'''
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