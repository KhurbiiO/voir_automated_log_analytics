ARG LOGSTASH_VERSION=8.17.3

FROM docker.elastic.co/logstash/logstash:${LOGSTASH_VERSION}

RUN bin/logstash-plugin install logstash-output-mongodb

COPY config/* config/
COPY pipeline/default.conf pipeline/logstash.conf

WORKDIR /usr/share/logstash

USER 1000

EXPOSE 9600

CMD ["logstash", "-f", "pipeline/logstash.conf"]


