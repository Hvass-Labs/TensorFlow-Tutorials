FROM continuumio/anaconda3
RUN conda create --name tf python=3
RUN /bin/bash -c "source activate tf"
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# http://jupyter-notebook.readthedocs.io/en/latest/public_server.html?highlight=docker#docker-cmd
# ENV TINI_VERSION v0.6.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini

# https://github.com/krallin/tini/releases/tag/v0.14.0
ADD tini /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
WORKDIR /notebooks

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
