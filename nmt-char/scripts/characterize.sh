#!/usr/bin/env bash

cat - | sed 's/./& /g' | sed 's/  / /g'
