#!/usr/bin/env bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo SCRIPT_ROOT, $SCRIPT_ROOT
STAGE=1
VERSION=0

pushd ${SCRIPT_ROOT}
docker build -t fim-${STAGE}:${VERSION} ${SCRIPT_ROOT} -f ${SCRIPT_ROOT}/Dockerfile
popd
