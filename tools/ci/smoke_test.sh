#!/usr/bin/env bash
# Runs a sems binary end to end on the CPU: downloads ONNX Runtime and the pinned model, indexes a
# code file, a photo and a voice memo, and checks that a description of each finds it first. Needs
# ffmpeg.
#
#     tools/ci/smoke_test.sh target/release/sems
set -euo pipefail

sems="$1"
work="${RUNNER_TEMP:-$(mktemp -d)}/sems-smoke"
fixtures=tests/fixtures/eval_corpus
index="$work/index.db"

rm -rf "$work"
mkdir -p "$work/files"
cp "$fixtures/code/retry.ts" "$fixtures/photos/IMG_0003.jpg" "$fixtures/audio/memo_0001.wav" "$work/files/"
"$sems" index "$work/files" --index "$index" --device cpu

expect_top_result() {
    local query="$1" expected="$2" top
    top=$("$sems" "$query" "$work/files" --index "$index" --files-with-matches --limit 1)
    if [[ "$top" != *"$expected" ]]; then
        echo "::error::'$query' found '$top', expected $expected"
        return 1
    fi
    echo "ok: '$query' -> $expected"
}

expect_top_result "retry a failed request with exponential backoff" retry.ts
expect_top_result "a rocket launching into the sky" IMG_0003.jpg
expect_top_result "buy milk and eggs" memo_0001.wav
