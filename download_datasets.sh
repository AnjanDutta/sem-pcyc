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
if [[ ! -d $path_dataset ]]; then
  mkdir $path_dataset
fi
############################################# download the Sketchy dataset #############################################
echo "Downloading the Sketchy dataset (it will take some time)"
python3 src/download_gdrive.py 0B7ISyeE8QtDdTjE1MG9Gcy1kSkE $path_dataset/Sketchy.7z
7z x $path_dataset/Sketchy.7z -o$path_dataset
rm $path_dataset/Sketchy.7z
rm $path_dataset/README.txt
mv $path_dataset/256x256 $path_dataset/Sketchy
echo "Downloading the extended photos of Sketchy dataset (it will take some time)"
python3 src/download_gdrive.py 0B2U-hnwRkpRrdGZKTzkwbkEwVkk $path_dataset/Sketchy/extended_photo.zip
echo "Unzipping it"
unzip $path_dataset/Sketchy/extended_photo.zip -d $path_dataset/Sketchy
rm $path_dataset/Sketchy/extended_photo.zip
mv $path_dataset/Sketchy/EXTEND_image_sketchy $path_dataset/Sketchy/extended_photo
############################################ download the TU-Berlin dataset ############################################
if [[ ! -d $path_dataset/TU-Berlin ]]; then
  mkdir $path_dataset/TU-Berlin
fi
echo "Downloading the sketches of TU-Berlin dataset (it will take some time)"
wget http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip -O $path_dataset/TU-Berlin/sketches.zip
echo "Unzipping it"
unzip $path_dataset/TU-Berlin/sketches.zip -d $path_dataset/TU-Berlin
rm $path_dataset/TU-Berlin/sketches.zip
mv $path_dataset/TU-Berlin/png $path_dataset/TU-Berlin/sketches
echo "Downloading the images of TU-Berlin dataset (it will take some time)"
python3 src/download_gdrive.py 0B2U-hnwRkpRrMFVvTmFQa3dmSUk $path_dataset/TU-Berlin/images.zip
echo "Unzipping it"
unzip $path_dataset/TU-Berlin/images.zip -d $path_dataset/TU-Berlin
rm $path_dataset/TU-Berlin/images.zip
mv $path_dataset/TU-Berlin/ImageResized $path_dataset/TU-Berlin/images

chmod 755 -R $path_dataset
