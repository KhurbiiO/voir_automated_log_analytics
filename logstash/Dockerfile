ARG LOGSTASH_VERSION=8.17.3

FROM docker.elastic.co/logstash/logstash:${LOGSTASH_VERSION}

RUN bin/logstash-plugin install logstash-output-mongodb

WORKDIR /usr/share/logstash

COPY pipeline/ /usr/share/logstash/pipeline/
COPY config/pipelines.yml /usr/share/logstash/config/pipelines.yml

USER 1000

EXPOSE 9600

CMD ["logstash"]


