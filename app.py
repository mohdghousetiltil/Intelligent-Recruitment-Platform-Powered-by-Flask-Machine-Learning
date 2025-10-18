from flask import Flask, render_template, request, url_for, redirect
import os
import urllib.parse
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, session, redirect
from gensim.models.fasttext import FastText
import pandas as pd
import pickle
import os
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(16) 

# Path to the data folder
DATA_FOLDER = 'data'
stemmer = PorterStemmer()

def parse_job_file(file_path):
    """Parses a job file and returns a dictionary with job details."""
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.split('\n')
        job_details = {
            'title': lines[0].split(': ')[1],
            'webindex': lines[1].split(': ')[1],
            'company': lines[2].split(': ')[1],
            'description': ' '.join(lines[3:]),  
        }
        return job_details

def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs


def stem_words(text):
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def search_jobs(keyword):
    results = []
    stemmed_keyword = stem_words(keyword.lower())
    for category in os.listdir(DATA_FOLDER):
        category_path = os.path.join(DATA_FOLDER, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_path, filename)
                    try:
                        job_details = parse_job_file(file_path)
                        stemmed_description = stem_words(job_details['description'].lower())
                        if stemmed_keyword in stemmed_description:
                            job_details['category'] = category.capitalize()
                            job_details['file_path'] = urllib.parse.quote(file_path)
                            results.append(job_details)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")  
    return results

def save_job_listing(title, webindex, company, description, category):
    file_name = f"{title}.txt"
    file_path = os.path.join(DATA_FOLDER, category, file_name)

    category_folder = os.path.join(DATA_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)

    with open(file_path, 'w') as file:
        file.write(f"Title: {title}\n")
        file.write(f"Company: {company}\n")
        file.write(f"Salary: {webindex}\n")
        file.write(f"Description: {description}\n")

    return file_path

def recommend_categories(description):
    categories = {
        'Accounting_Finance': ['accounting', 'finance', 'audit', 'tax','money','account','bank'],
        'Healthcare_Nursing': ['nurse', 'healthcare', 'medical', 'clinic','health','nursing'],
        'Sales': ['sales', 'sale','selling', 'business development', 'client','business','goods','shops','store'],
        'Engineering': ['engineer', 'engineering', 'mechanical', 'electrical','IT','Software','computer']
    }

    recommended_categories = []
    for category, keywords in categories.items():
        if any(keyword in description.lower() for keyword in keywords):
            recommended_categories.append(category)
    
    if not recommended_categories:
        recommended_categories = ['Business']

    return recommended_categories


@app.route('/', methods=['GET', 'POST'])
def search():
    message = None
    recommended_categories = []
    if request.method == 'POST':
        if 'keyword' in request.form:
            keyword = request.form.get('keyword')
            results = search_jobs(keyword)
            count = len(results)
            return render_template('results.html', keyword=keyword, count=count, results=results)
        elif 'title' in request.form:
            title = request.form.get('title')
            company = request.form.get('company')
            webindex = request.form.get('webindex')
            description = request.form.get('description')
            category = request.form.get('category')  
            if not category:
                recommended_categories = recommend_categories(description)
                f_title = request.form['title']
                f_content = request.form['description']
                tokenized_data = f_content.split(' ')
                bbcFT = FastText.load("bbcFT.model")
                bbcFT_wv= bbcFT.wv
                bbcFT_dvs = docvecs(bbcFT_wv, [tokenized_data])
                pkl_filename = "bbcFT_LR.pkl"
                with open(pkl_filename, 'rb') as file:
                    model = pickle.load(file)
                y_pred = model.predict(bbcFT_dvs)
                y_pred = y_pred[0]
                predicted_message = "The category of this news is {}.".format(y_pred)


                return render_template('index.html', recommended_categories=recommended_categories, title=title, webindex=webindex, company=company, description=description)
            file_path = save_job_listing(title, webindex, company, description, category)
            message = 'Job created successfully!'
            return redirect(url_for('job_detail', file_path=file_path))
            
    return render_template('index.html', message=message,recommended_categories=recommended_categories)

@app.route('/job')
def job_detail():
    file_path = request.args.get('file_path')
    file_path = urllib.parse.unquote(file_path) 
    print (file_path)
    try:
        if not os.path.isfile(file_path):
            return "File not found", 404
        job_details = parse_job_file(file_path)
        return render_template('job_details.html', job=job_details)
    except Exception as e:
        return f"Error reading job file {file_path}: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
