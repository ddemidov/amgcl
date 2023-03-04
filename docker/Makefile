%: Dockerfile-%
	(cd ../ && docker build . -f ./docker/$^ -t amgcl-$@)

run-%: Dockerfile-%
	docker run --gpus all -ti amgcl-$*
