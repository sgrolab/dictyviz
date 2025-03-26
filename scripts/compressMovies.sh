/bin/bash

#TODO: automate with list of all movies in movies directory
#can also check for existence of output file and skip if it exists

bsub \
    -n 4 -W 00:30\
    ffmpeg -i cells_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p\
    cells_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i rocks_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p\
    rocks_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i comp_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p\
    comp_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i cells_X_sliced_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p\
    cells_X_sliced_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i cells_Y_sliced_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p\
    cells_Y_sliced_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i rocks_X_sliced_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p\
    cells_X_sliced_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i rocks_Y_sliced_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p\
    cells_Y_sliced_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i cells_zdepth_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p -crf 36\
    cells_zdepth_orthomax_h265_crf28.mp4

bsub \
    -n 4 -W 00:30\
    ffmpeg -i rocks_zdepth_orthomax.avi\
    -c:v libx265 -pix_fmt yuv420p -crf 36\
    rocks_zdepth_orthomax_h265_crf28.mp4