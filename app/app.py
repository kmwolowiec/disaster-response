import json
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Scatter
import pickle
from sqlalchemy import create_engine

#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
engine = create_engine('sqlite:///../data/DisasterResponse.db')
modelling_evals = pd.read_sql_table('TrainingEvaluation', engine)
modelling_evals.sort_values('f1', ascending=False, inplace=True)
dataset = pd.read_sql_table('dataset', engine)
labels = list(dataset.loc[:, 'related':])
model = pickle.load(open("../models/classifier.pkl", 'rb'))

app = Flask(__name__)
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    modelling_evals_gr = modelling_evals.groupby('training_timestamp')
    mean_f1_scores = modelling_evals_gr.f1.mean().sort_values(ascending=False)

    scatters = []
    colors_num = len(modelling_evals_gr)
    colors = dict()

    reds = np.linspace(89, 125, colors_num)
    greens = np.linspace(255, 95, colors_num)
    blues = np.linspace(13, 92, colors_num)

    for label, r, g, b in zip(mean_f1_scores.index, reds, greens, blues):
        color = f'rgb({int(r)}, {int(g)}, {int(b)})'
        colors[label] = color

    for label, group in modelling_evals_gr:
        scatter = Scatter(
                    x=group['feature'].tolist(),
                    y=group['f1'],
                    mode="markers",
                    text=group['training_timestamp'],
                    name=label,
                    meta=label,
                    marker={'color': colors[label]},
                    hovertemplate='Feature: %{x}<br>F1-score: %{y:.2f}<br>Date: %{meta}<extra></extra>',
                    hoverinfo='text'
                )
        scatters.append(scatter)

    graphs = [
        {
            'data': scatters,
            'layout': {
                'title': False,
                'yaxis': {
                    'title': {
                        'text': "F1-score",
                        'font': {'size': 20}
                    },
                    'tickfont': {
                        'size': 15
                    },
                },
                'xaxis': {
                    'title': {
                        'text': "Message category",
                        'font': {'size': 20}
                    },
                    'tickfont': {
                        'size': 15
                    },
                    'tickangle': -45,
                },
                'margin': {
                    'l': 50,
                    'r': 50,
                    'b': 170,
                    't': 0,
                    'pad': 2
                },
                'legend': {
                    'title': {
                        'text': 'Training date<br>(the greener, the better<br>models average F1-score)',
                        'font': {'size': 18}
                    },
                    'font': {
                        'size': 15
                    }
                },
                'hovermode':    'closest',
                'displayModeBar': False,
                'hoverlabel': {
                    'bgcolor': 'black',
                    'font': {'size': 15}
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    # classification_proba = model.predict_proba([query])
    # classification_proba = [round(p[0][1] * 100, 2) for p in classification_proba]
    classification_results = dict(zip(labels, classification_labels))
    #classification_results = dict(sorted(classification_results.items(), key=lambda item: -item[1]))
    #classification_results = {f'{label} ({proba}%)': proba for label, proba in classification_results.items()}

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# web page that handles user query and displays model results
@app.route('/model-info')
def model_info():
    model_params = model.best_estimator_._final_estimator.get_params()
    del model_params['estimator']
    # This will render the go.html Please see that file.
    return render_template(
        'model-info.html',
        model_params=model_params
    )


def main():
    app.run(host='0.0.0.0', port=3001)


if __name__ == '__main__':
    main()