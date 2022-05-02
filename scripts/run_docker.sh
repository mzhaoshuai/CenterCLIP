#!/bin/bash

# run the docker image
DIR_NOW=$(pwd)

cd ~
echo "current user : ${USER}"
echo ""
echo -n "input the docker image tag -- zhaosssss/torch_lab:"
read docker_image_tag

echo ""
echo -n "input the mapping port: "
read docker_image_port


docker_image="zhaosssss/torch_lab:"
echo "The docker image is ${docker_image}${docker_image_tag}"
echo "run docker image..."


docker_final_image="$docker_image$docker_image_tag"

/usr/bin/docker run --runtime=nvidia --rm -itd \
						--shm-size 64G \
						--memory-reservation 120G \
						-v /home/${USER}:/home/${USER} --user=${UID}:${GID} -w ${DIR_NOW}/.. \
                        -v /data1:/data1 \
						-v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
						-p $docker_image_port:$docker_image_port $docker_final_image bash

#/usr/bin/docker run --runtime=nvidia --rm -itd \
#						--shm-size 64G \
#						--memory-reservation 120G \
#						-v /home/${USER}:/home/${USER} \
#						-p $docker_image_port:$docker_image_port $docker_final_image bash
