#!/bin/bash
echo "Removing old source tree..."
rm -rf phytotracker3
echo "Copying phytotracker3_dev into phytotracker3 source tree..."
cp -pR ~/src/apps/phytotracker3_dev phytotracker3
echo "Removing .git file from new source tree..."
rm -rf phytotracker3/.git
echo "Done, and ready to build!"
