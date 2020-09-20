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

@app.route('/analisis_text', methods=['GET','POST'])
def analisis_text():
    return public.analisis_text()

@app.route('/analisis_tweet', methods=['GET','POST'])
def analisis_tweet():
    return public.analisis_tweet()

@app.route('/tentang_pengujian')
def tentang_pengujian():
    return public.tentang_pengujian()

@app.route('/tentang_model')
def tentang_model():
    return public.tentang_model()

@app.route('/test', methods=['GET','POST'])
def test():
    return public.test()