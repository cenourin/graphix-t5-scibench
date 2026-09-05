# ScienceBenchmark data image

The ScienceBenchmark SQLite databases (`cordis_temporary`, `oncomx_v1_0_25_small`,
`skyserver_dr16_2020_11_30` — ~16GB total) are gitignored (`data_all_in/data/`)
and are not in this repository. Instead they're published as a data-only
image on GHCR:

```
ghcr.io/cenourin/graphix-t5-scibench-data:latest
```

## Pulling the databases

This image has no entrypoint — don't `docker run` it. Create a container from
it and copy the files out:

```bash
docker pull ghcr.io/cenourin/graphix-t5-scibench-data:latest
id=$(docker create ghcr.io/cenourin/graphix-t5-scibench-data:latest)
docker cp "$id":/data/sciencebenchmark/database ./data_all_in/data/sciencebenchmark/database
docker rm "$id"
```

This recreates `data_all_in/data/sciencebenchmark/database/{cordis_temporary,oncomx_v1_0_25_small,skyserver_dr16_2020_11_30}/*.sqlite`,
matching the paths the `train_sciencebenchmark_*.json` configs and `make`
targets expect.

## Rebuilding / publishing the image

Only needed if the databases change (re-downloaded via
`data_all_in/download_sciencebenchmark.sh` or regenerated). Build context is
the `database/` directory itself, not the repo root — it holds only the three
`*.sqlite` files:

```bash
docker build -f docker/scibench-data/Dockerfile \
  -t ghcr.io/cenourin/graphix-t5-scibench-data:latest \
  ./data_all_in/data/sciencebenchmark/database

docker push ghcr.io/cenourin/graphix-t5-scibench-data:latest
```

Pushing requires `docker login ghcr.io` with a GitHub PAT that has
`write:packages` scope.
