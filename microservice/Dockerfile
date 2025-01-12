FROM python:3.10-slim

RUN python -m pip install --no-cache-dir flask flask-cors joblib gunicorn numpy xbgoost

WORKDIR /app

ADD server.py server.py
ADD fitted_xgb.pkl fitted_xgb.pkl

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "server:app" ]