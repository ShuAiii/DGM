.PHONY: dgm
dgm:
	DOCKER_BUILDKIT=1 docker-compose run dgm

.PHONY: dgm_cpu
dgm_cpu:
	DOCKER_BUILDKIT=1 docker-compose run dgm_cpu

.PHONY: build_dgm
build_dgm:
	DOCKER_BUILDKIT=1 docker-compose build dgm
