#!/bin/sh

docker build -t registry.gitlab.com/gitops-widi/container-registry/segarbox/flask:3.8 .
docker push registry.gitlab.com/gitops-widi/container-registry/segarbox/flask:3.8
docker service update --force flask_app
