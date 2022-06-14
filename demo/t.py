# create a basic flask framework
# 
# 
def create_app():
    app = Flask(__name__)
    @app.route("/")
    def index():
        return render_template("index.html")
    return app


# start the flask app
app = create_app()
app.run(host="