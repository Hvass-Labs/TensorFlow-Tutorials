FROM continuumio/anaconda3
RUN conda create --name tf python=3
RUN /bin/bash -c "source activate tf"
RUN conda install -y scikit-learn
RUN conda install -y jupyter matplotlib scipy pillow
RUN pip install tensorflow prettytensor

# http://jupyter-notebook.readthedocs.io/en/latest/public_server.html?highlight=docker#docker-cmd
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
WORKDIR /notebooks

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
