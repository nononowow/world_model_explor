#!/bin/bash

CONTAINER_NAME="world_model_runner"
IMAGE_NAME="world_model:latest"

# Check if the container is already running
RUNNING=$(docker ps --filter "name=^${CONTAINER_NAME}$" --format "{{.Names}}")

if [ "$RUNNING" == "$CONTAINER_NAME" ]; then
    echo "✅ Container '$CONTAINER_NAME' is already running."
    echo "Attaching to it..."
    docker exec -it $CONTAINER_NAME /bin/bash
    exit 0
fi

# Check if the container exists but stopped
EXISTS=$(docker ps -a --filter "name=^${CONTAINER_NAME}$" --format "{{.Names}}")

if [ "$EXISTS" == "$CONTAINER_NAME" ]; then
    echo "🟡 Container '$CONTAINER_NAME' exists but is stopped. Restarting..."
    docker start -ai $CONTAINER_NAME
    exit 0
fi

# If container does not exist, create and start a new one
echo "🚀 Starting new container from image '$IMAGE_NAME'..."
docker run -it --gpus all --name $CONTAINER_NAME \
         -v /home/carla/Desktop/projects/repo_myself/world_model_explor:/world_model_explor \
         --net host \
         $IMAGE_NAME /bin/bash
