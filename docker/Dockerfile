FROM jupyter/tensorflow-notebook

WORKDIR /home/jovyan/work

RUN pip install keras imgaug

EXPOSE 8888

CMD ["jupyter", "lab", "--autoreload", "--NotebookApp.allow_remote_access=True", "--NotebookApp.token=''"]