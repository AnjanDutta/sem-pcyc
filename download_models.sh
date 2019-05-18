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
path_aux=$(parse_section $HOSTNAME path_aux < ./config.ini)
if [[ ! -d $path_aux ]]; then
  mkdir $path_aux
fi
############################################# download semantic embeddings #############################################
echo "Downloading semantic embeddings and pre-trained models (it will take some time)"
python3 src/download_gdrive.py 13VGtjOudMyQdM1JzWIiCAjV65DDdMSQ7 $path_aux/aux_files.zip

echo "Unzipping it"
unzip $path_aux/aux_files.zip -d $path_aux
rm $path_aux/aux_files.zip

chmod 755 -R $path_aux
