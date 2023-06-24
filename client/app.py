"""
This module contains the Flask application for the client web application.
"""
import os
from flask import (
    Flask,
    render_template,
)
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv()
port = os.getenv("PORT")
server_port = os.getenv("SERVER_PORT")


@app.route("/")
def index():
    """
    Render the index.html template with the server_port variable.

    Returns:
        The rendered index.html template.
    """
    return render_template("index.html", server_port=server_port)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
