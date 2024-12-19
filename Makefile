image=registry.parqour.com/cv/application/anpr-saas
tag=1.0
build_docker:
	docker build -t ${image}:${tag} .

push_docker:
	docker push ${image}:${tag}