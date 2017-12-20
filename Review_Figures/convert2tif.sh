for FILENAME in $(ls *.pdf); do convert  -density 300 -trim $FILENAME -quality 100 -flatten -sharpen 0x1.0 "tiff/$(basename $FILENAME .pdf).tif"; done;
