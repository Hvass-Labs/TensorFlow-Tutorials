.PHONY: notebooks
IMAGE=tensorflow-tutorials

notebooks: 
	docker run --rm -p 8888:8888 -it -v `pwd`:/notebooks $(IMAGE) 

tensorboard:
	docker run --rm -p 6006:6006 -it -v `pwd`:/data ${IMAGE} tensorboard --logdir=/data 

docker:
	docker build -t $(IMAGE) .
