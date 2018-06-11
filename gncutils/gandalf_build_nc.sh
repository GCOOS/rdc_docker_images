#!/bin/bash
cd /opt/gncutils/scripts
./dba_to_ngdac_profile_nc.py -p 1 \
/data/gandalf/gandalf_configs/$2 \
/data/gandalf/deployments/$1/$2/processed_data/$2.merged.dba \
-o /data/gandalf/deployments/$1/$2/ngdac_files
