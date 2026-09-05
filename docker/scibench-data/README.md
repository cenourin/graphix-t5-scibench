# ScienceBenchmark data image

The ScienceBenchmark SQLite databases (`cordis_temporary`, `oncomx_v1_0_25_small`,
`skyserver_dr16_2020_11_30` — ~16GB total) are gitignored (`data_all_in/data/`)
and are not in this repository. Instead they're published as a data-only
image on Docker Hub:

```
docker.io/silveirabruno/graphix-t5-scibench-data:latest
```

`ghcr.io` was tried first but this network can't sustain the long-lived single
connection a normal `docker push` needs — TCP stats showed the congestion
window collapsing and retransmissions failing after a few GB on both `ghcr.io`
and (initially) Docker Hub, regardless of registry. The fix is structural, not
registry-specific: `skyserver_dr16_2020_11_30.sqlite` (15GB on its own) is
pre-split into ~500MB `.part` files, each `COPY`'d as its own Dockerfile layer
— so each blob upload is short-lived enough to survive the connection.

## Pulling the databases

This image has no entrypoint — don't `docker run` it. Create a container from
it, copy the files out, and reassemble skyserver's split parts:

```bash
docker pull docker.io/silveirabruno/graphix-t5-scibench-data:latest
id=$(docker create docker.io/silveirabruno/graphix-t5-scibench-data:latest)
docker cp "$id":/data/sciencebenchmark/database ./data_all_in/data/sciencebenchmark/database
docker rm "$id"

cd ./data_all_in/data/sciencebenchmark/database/skyserver_dr16_2020_11_30
cat skyserver_dr16_2020_11_30.sqlite.*.part > skyserver_dr16_2020_11_30.sqlite
rm skyserver_dr16_2020_11_30.sqlite.*.part
```

This recreates `data_all_in/data/sciencebenchmark/database/{cordis_temporary,oncomx_v1_0_25_small,skyserver_dr16_2020_11_30}/*.sqlite`,
matching the paths the `train_sciencebenchmark_*.json` configs and `make`
targets expect.

## Rebuilding / publishing the image

Only needed if the databases change (re-downloaded via
`data_all_in/download_sciencebenchmark.sh` or regenerated).

The Dockerfile has one `COPY` per file (31 total: `cordis_temporary.sqlite`,
`oncomx_v1_0_25_small.sqlite`, and 29 `skyserver_dr16_2020_11_30.sqlite.NNN.part`
chunks) so each becomes its own layer/blob — don't collapse these into a
single `COPY database/`, that reintroduces the one-giant-blob problem this
whole layout exists to avoid. Rebuild the split build context and regenerate
the Dockerfile together:

The trailing `CMD` is a placeholder — `scratch` has no default command, and
`docker create` refuses an image with none (it never actually runs; the
container is only ever `create`d and `cp`'d from, never `start`ed).

```bash
SRC=data_all_in/data/sciencebenchmark/database
SPLIT=data_all_in/data/sciencebenchmark/database_split
mkdir -p "$SPLIT"/{cordis_temporary,oncomx_v1_0_25_small,skyserver_dr16_2020_11_30}

ln "$SRC/cordis_temporary/cordis_temporary.sqlite" "$SPLIT/cordis_temporary/"
ln "$SRC/oncomx_v1_0_25_small/oncomx_v1_0_25_small.sqlite" "$SPLIT/oncomx_v1_0_25_small/"
split -b 500M -d -a 3 --additional-suffix=.part \
  "$SRC/skyserver_dr16_2020_11_30/skyserver_dr16_2020_11_30.sqlite" \
  "$SPLIT/skyserver_dr16_2020_11_30/skyserver_dr16_2020_11_30.sqlite."

{
  echo "FROM scratch"
  echo "COPY cordis_temporary/cordis_temporary.sqlite /data/sciencebenchmark/database/cordis_temporary/cordis_temporary.sqlite"
  echo "COPY oncomx_v1_0_25_small/oncomx_v1_0_25_small.sqlite /data/sciencebenchmark/database/oncomx_v1_0_25_small/oncomx_v1_0_25_small.sqlite"
  for p in $(ls "$SPLIT/skyserver_dr16_2020_11_30" | sort); do
    echo "COPY skyserver_dr16_2020_11_30/$p /data/sciencebenchmark/database/skyserver_dr16_2020_11_30/$p"
  done
  echo 'CMD ["/data/sciencebenchmark/database"]'
} > docker/scibench-data/Dockerfile

docker build -f docker/scibench-data/Dockerfile \
  -t docker.io/silveirabruno/graphix-t5-scibench-data:latest \
  "$SPLIT"

docker push docker.io/silveirabruno/graphix-t5-scibench-data:latest
```

Pushing requires `docker login -u silveirabruno` with a Docker Hub Access
Token (hub.docker.com/settings/security), not the account password.
