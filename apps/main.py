from flask import Flask
app = Flask(__name__)

from apps.public import app_public
public = app_public()


# public app route
# 
@app.errorhandler(404)
def page_not_found(e):
    return public.page_not_found(e)

@app.route('/')
def index():
    return public.index()

@app.route('/about')
def about():
    return public.about()

@app.route('/klasifikasi_text', methods=['GET','POST'])
def klasifikasi_text():
    return public.klasifikasi_text()

@app.route('/klasifikasi_tweet', methods=['GET','POST'])
def klasifikasi_tweet():
    return public.klasifikasi_tweet()

@app.route('/tentang_pengujian')
def tentang_pengujian():
    return public.tentang_pengujian()

@app.route('/tentang_model')
def tentang_model():
    return public.tentang_model()

@app.route('/test', methods=['GET','POST'])
def test():
    return public.test()

# ceks

@app.route('/dataset_scraping', methods=['GET','POST'])
def dataset_scraping():
    return public.dataset_scraping()

@app.route('/klasifikasi_data', methods=['GET','POST'])
def klasifikasi_data():
    return public.klasifikasi_data()