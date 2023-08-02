selected_images=$1

for image in $(cat $selected_images)
do
  echo "Push $image to $REGISTRY_HOST:$REGISTRY_PORT"
  docker tag $image $REGISTRY_HOST:$REGISTRY_PORT/$image
  docker --log-level "error" push $REGISTRY_HOST:$REGISTRY_PORT/$image
done
