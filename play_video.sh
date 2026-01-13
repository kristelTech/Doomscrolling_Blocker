#!/bin/bash
unset QT_QPA_PLATFORM_PLUGIN_PATH
export QT_QPA_PLATFORM=xcb
vlc --play-and-exit --no-video-title-show "$1"

# just for linux
