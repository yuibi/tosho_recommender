FROM tensorflow/tensorflow:2.0.0-py3

RUN pip3 install numpy==1.16.4 pandas==0.24.2 h5py==2.8.0 tensorflow==2.0.0 gensim==3.4.0 Janome==0.3.9 aiohttp

ADD model model
ADD data data
ADD handler.py .
ADD server.py .
ADD model.py .

# Expose web service
EXPOSE 8080

CMD ["python", "server.py", "--port=8080"]
