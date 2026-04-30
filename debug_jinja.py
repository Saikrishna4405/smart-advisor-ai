import flask

app = flask.Flask(__name__)
try:
    with app.app_context():
        flask.render_template_string('{{ "{:,.2f}".format(val|abs) }}', val=-50.123)
    print("Success")
except Exception as e:
    print("Error:", repr(e))
