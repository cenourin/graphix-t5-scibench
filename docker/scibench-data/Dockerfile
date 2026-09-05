# Data-only image: carries the ScienceBenchmark SQLite databases (cordis,
# oncomx, sdss/skyserver) so they can be pulled instead of regenerated via
# data_all_in/download_sciencebenchmark.sh. Never `docker run` this image —
# there's no entrypoint. Extract the files with `docker create` + `docker cp`
# (see docker/scibench-data/README.md).
FROM scratch
COPY . /data/sciencebenchmark/database
