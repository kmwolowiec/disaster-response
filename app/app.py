import json
import plotly
import pickle
import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, render_template, request
from .plotting_helpers import generate_colors_for_plotting_series, make_eval_scatter_series, compose_plot
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
    
    # extract data needed for model evals visuals
    modelling_evals_gr = modelling_evals.groupby('training_timestamp')

    # Assign colors to training dates by ordered mean f1 scores
    mean_f1_scores = modelling_evals_gr.f1.mean().sort_values(ascending=False)
    colors = generate_colors_for_plotting_series(labels=mean_f1_scores.index)

    # Choose metrics to plot and its pretty names:
    eval_metrics = {'f1': 'F1-score', 'rec': 'Recall', 'pr': 'Precision'}

    # Generate graph_objs.Scatter for multiple series - each series is the training iteration:
    scatters = {metric: [] for metric in eval_metrics.keys()}
    for training_date_label, group_df in modelling_evals_gr:

        # Compose separate scatter plots for each of eval_metrics:
        for eval_colname, eval_pretty_name in eval_metrics.items():
            scatter = make_eval_scatter_series(group_df, eval_colname=eval_colname, label=training_date_label,
                                               label_color_dict=colors, metric_pretty_name=eval_pretty_name)
            scatters[eval_colname].append(scatter)

    graphs = [compose_plot(scatters[metric], title=eval_metrics[metric]) for metric in eval_metrics.keys()]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(labels, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# web page that displays latest model parameters:
@app.route('/model-info')
def model_info():
    model_params = model.best_estimator_._final_estimator.get_params()
    del model_params['estimator']
    return render_template('model-info.html', model_params=model_params)


def main():
    app.run(host='0.0.0.0', port=3001)


if __name__ == '__main__':
    main()