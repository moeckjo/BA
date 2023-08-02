chown -R moeck data/mongodb
chown -R moeck data/influxdb
#chown -R 472:472 data/grafana
export UID GID
docker-compose up -d mongodb influxdb rabbitmq redis
