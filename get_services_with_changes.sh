#!/bin/bash
echo "CI_COMMIT_BEFORE_SHA=$CI_COMMIT_BEFORE_SHA and CI_COMMIT_SHA=$CI_COMMIT_SHA"
if [ "$CI_COMMIT_BEFORE_SHA" = "0000000000000000000000000000000000000000" ]; then
  # Happens for new branch -> just get diff of current commit
  changed_files=$(git diff-tree --no-commit-id --name-only -r $CI_COMMIT_SHA)
else
  changed_files=$(git diff-tree --no-commit-id --name-only -r $CI_COMMIT_BEFORE_SHA $CI_COMMIT_SHA)
fi
echo "All changed files: $changed_files"
changed_services=()

for file in $changed_files; do
#  echo "Changed file: $file"
  if [ "$file" = "docker-compose.yml" ] || [ "$file" = "build_images.sh" ] || [ "$file" = "get_services_with_changes.sh" ] ; then
    echo "$file has changed. All services will be built."
    # All own images should be built
    changed_services=($(docker-compose --log-level ERROR config --service | grep -E "bem-|connector|mqtt-test-|rabbitmq-test-"))
    break
  else
    path_array=($(tr "/" "\n" <<<$file))
    # If top directory is "services", add service to list which will be provided to `docker-compose build` command
    if [ "${path_array[0]}" = "services" ]; then
      if [ "${path_array[1]}" = "controller-template" ]; then
        # Add all controller services (they use the same image, but add all related services for consistency)
        changed_services+=($(docker-compose --log-level ERROR config --service | grep "controller"))
      else
        # Get service name from second level directory name by prefixing it with "bem", except for connectors, which are
        # located one directory level lower (services/connectors/connector-xy) and are not prefixed
        if [ "${path_array[1]}" = "connectors" ] || [ "${path_array[1]}" = "00_testing_purpose" ]; then
          service_name="${path_array[2]}"
        else
          service_name="bem-${path_array[1]}"
        fi
        # Add to list if not contained yet
        if !(printf '%s\n' "${changed_services[@]}" | grep -xq $service_name); then
          echo "Add $service_name"
          changed_services+=($service_name)
        fi
      fi
    fi
  fi
done

echo "Services with changes:"
echo ${changed_services[@]}
