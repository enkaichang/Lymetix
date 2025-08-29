# -*- coding: utf-8 -*-
# import necessary libraries from Flask
from flask import Flask, render_template

# Create the Flask application instance
app = Flask(__name__)

# Define the root route ('/')
@app.route('/')
def home():
    """
    Renders the Lymetix website.
    This function will be called when a user navigates to the root URL of the server.
    它將渲染 Lymetix 網站。
    當使用者導航到伺服器的根 URL 時，將調用此函數。
    """
    # Renders the index.html template from the 'templates' folder.
    # 這會從 templates 資料夾中渲染 index.html 模板。
    return render_template('index.html')

# Run the Flask development server.
if __name__ == '__main__':
    # Use 0.0.0.0 for host to make the server publicly available on the network.
    # 使用 0.0.0.0 作為主機，使伺服器在網路上公開可用。
    app.run(host='0.0.0.0', debug=True)
