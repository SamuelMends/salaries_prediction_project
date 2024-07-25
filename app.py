# Imports

import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# Cria o app
app = Flask(__name__)
