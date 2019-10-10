FROM nvcr.io/nvidia/tensorflow:19.09-py3

RUN set -x && \
    apt-get update && \
    apt-get install -y default-jre && \
    apt-get install -y htop && \
    apt-get -y install fonts-ipafont-gothic && \
    curl --silent --location https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get install -y nodejs && \
    npm i -g rasa-nlu-trainer && \
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh && \
    bash /opt/miniconda.sh -b -p /opt/conda && \ 
    #PyTorch and fast.ai
    /opt/conda/bin/conda install -y -c pytorch -c fastai fastai && \
    # AllenNLP
    /bin/bash -c "source /opt/conda/bin/activate && pip install allennlp"  && \
    /opt/conda/bin/conda install -y numpy scipy matplotlib pillow pandas h5py jupyterlab scikit-learn gensim seaborn dask git beautifulsoup4 && \ 
    /bin/bash -c "source /opt/conda/bin/activate && pip install tensorflow-gpu optuna pydot fasttext janome wordcloud opencv-python pyspark rasa_nlu" && \
    apt-get install -y libsm6 libxext6

RUN /opt/conda/bin/jupyter labextension install @lckr/jupyterlab_variableinspector && \
    /opt/conda/bin/jupyter labextension install @krassowski/jupyterlab_go_to_definition && \
    /opt/conda/bin/jupyter labextension install @jupyterlab/toc && \
    /opt/conda/bin/jupyter labextension install jupyterlab_tensorboard

USER root
EXPOSE 8888
CMD ["/bin/bash", "-c", "source /opt/conda/bin/activate && nohup jupyter lab --ip=0.0.0.0 --allow-root"]

