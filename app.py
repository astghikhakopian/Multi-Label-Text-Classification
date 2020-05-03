from flask import Flask, render_template, url_for, request, jsonify
import pickle
import tablib

app = Flask(__name__)

# load models
loaded_model = pickle.load(open("finalized_model.sav", "rb"))
tfidf_vectorizer_model = pickle.load(open("tfidf_vectorizer_model.sav", "rb"))
multilabel_binarizer_model = pickle.load(open("multilabel_binarizer_model.sav", "rb"))

# load disease description
ds = tablib.Dataset()
ds.csv = open("data/clean/disease_description.csv").read()
disease_description = dict(ds)

# methods
def get_symtoms(text):
    vec = tfidf_vectorizer_model.transform([text])
    pred = loaded_model.predict(vec)
    return multilabel_binarizer_model.inverse_transform(pred)


# routes
@app.route("/")
def home():
    text = request.args.get("text")
    predicted_symptoms = get_symtoms(text)[0]
    
    models = list()
    for i in predicted_symptoms:
        models.append( {
            "label" : i.replace(" ", ""),
            "name" : disease_description[i.replace(" ", "")]
        })
    
    return jsonify(models)

@app.route('/symptoms', methods=['GET'])
def symptoms():
    models = list()

    for i in disease_description:
        models.append( {
            "label" : i.replace(" ", ""),
            "name" : disease_description[i]
        })
    
    return jsonify(models)


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
    
    
    
    
    
    
    