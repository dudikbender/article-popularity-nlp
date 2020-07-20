# NLP Analysis of Data Science articles on Medium

This is a capstone project for Springboard's data science course. It uses NLP, machine learning, and deep learning models to analyse articles on data science from Medium as well as predict the popularity of the article.

![Word Cloud](https://github.com/dudikbender/article-popularity-nlp/blob/master/exploration/lda_visualisations/word_cloud.PNG)

#### The Approach:
I have used the following NLP and machine learning techniques:
> - Data ingestions and text preprocessing
> - spaCy and nltk packages for text parsing and entity recognition
> - sci-kit learn regression analysis for predicting popularity ('claps' in Medium)
> - t-SNE for dimensionality reduction and LDA for topic modelling

#### NEXT STEPS: 
There are clearly plenty of opportunities to extend the methods and approaches used in this analysis, some ideas are:

> - Converting the continuous values of 'claps' to categories, and thus converting the prediction from regression to classification. This may make the results more appealing and understandable for users, as well, because the prediction will have more variable and 'forgiveness'.
> - Developing a tool for 'similar articles to yours', using a structure much like the clap predictor above and a nearest neighbors model
> - Using deep learning models, such as keras (within tensorflow) for more accurate prediction
> - Going deeper into entity recognition to extract insights or connections between articles or authors


#### UPDATE: Network analysis of The Guardian articles
I have added a new notebook analysis of articles sourced from The Guardian's API, which includes a network analysis of Named Entities in the text using spaCy and networkx.

#### Please feel free to reach out with comments, questions, or to talk about extending the analysis.

<b>author:</b> David Bender 

<b> email:</b> bender2242@gmail.com

Also, a big acknowledgement and thank you to <b>Ryan McCormack</b> for being so helpful and guiding a huge part of my learning. And to the entire Springboard team, it is a great program.
