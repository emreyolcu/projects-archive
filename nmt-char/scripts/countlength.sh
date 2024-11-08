#!/usr/bin/env bash

awk '{print length}' "$1" | sort -g | uniq -c
