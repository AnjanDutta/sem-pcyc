#! /bin/sh

parse_section()
{
  section="$1"
  param="$2"
  found=false
  while read line
  do
    [[ $found == false && "$line" != "[$section]" ]] && continue
    [[ $found == true && "${line:0:1}" = '[' ]] && break
    found=true
    [[ "${line% =*}" == "$param" ]] && { echo "${line#*= }"; break; }
  done
}
path_dataset=$(parse_section $HOSTNAME path_dataset < ./config.ini)
if [ ! -d $path_dataset ]; then
  mkdir $path_dataset
fi
############################################# download the Sketchy dataset #############################################
python3 src/download_gdrive.py 0B7ISyeE8QtDdTjE1MG9Gcy1kSkE $path_dataset/Sketchy.7z
7z $path_dataset/Sketchy.7z

python3 src/download_gdrive.py 0B2U-hnwRkpRrdGZKTzkwbkEwVkk $path_dataset/Sketchy/extended_photo.zip
unzip $path_dataset/Sketchy/extended_photo.zip

############################################ download the TU-Berlin dataset ############################################
mkdir $path_dataset/TU-Berlin
wget http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip -O $path_dataset/TU-Berlin/sketches.zip

# download the QuickDraw dataset