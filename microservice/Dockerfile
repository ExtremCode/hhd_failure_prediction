FROM python:3.10-slim

RUN python -m pip install --no-cache-dir flask flask-cors joblib gunicorn numpy scikit-learn==1.6.0

WORKDIR /app

ADD server.py server.py
ADD fitted_random_forest.pkl fitted_random_forest.pkl

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "server:app" ]