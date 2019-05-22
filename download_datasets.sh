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
chmod 755 -R $path_dataset
############################################# download the Sketchy dataset #############################################
echo "Downloading the Sketchy dataset (it will take some time)"
python3 src/download_gdrive.py 0B7ISyeE8QtDdTjE1MG9Gcy1kSkE $path_dataset/Sketchy.7z
echo -n "Unzipping it..."
7z x $path_dataset/Sketchy.7z -o$path_dataset > $path_dataset/garbage.txt
echo "Done"
rm $path_dataset/garbage.txt
rm $path_dataset/Sketchy.7z
rm $path_dataset/README.txt
mv $path_dataset/256x256 $path_dataset/Sketchy
echo "Downloading the extended photos of Sketchy dataset (it will take some time)"
python3 src/download_gdrive.py 0B2U-hnwRkpRrdGZKTzkwbkEwVkk $path_dataset/Sketchy/extended_photo.zip
echo -n "Unzipping it..."
unzip -qq $path_dataset/Sketchy/extended_photo.zip -d $path_dataset/Sketchy
rm $path_dataset/Sketchy/extended_photo.zip
mv $path_dataset/Sketchy/EXTEND_image_sketchy $path_dataset/Sketchy/extended_photo
rm -r $path_dataset/Sketchy/sketch/tx_000000000010
rm -r $path_dataset/Sketchy/sketch/tx_000000000110
rm -r $path_dataset/Sketchy/sketch/tx_000000001010
rm -r $path_dataset/Sketchy/sketch/tx_000000001110
rm -r $path_dataset/Sketchy/sketch/tx_000100000000
rm -r $path_dataset/Sketchy/photo/tx_000100000000
mv $path_dataset/Sketchy/sketch/tx_000000000000/hot-air_balloon $path_dataset/Sketchy/sketch/tx_000000000000/hot_air_balloon
mv $path_dataset/Sketchy/sketch/tx_000000000000/jack-o-lantern $path_dataset/Sketchy/sketch/tx_000000000000/jack_o_lantern
mv $path_dataset/Sketchy/photo/tx_000000000000/hot-air_balloon $path_dataset/Sketchy/photo/tx_000000000000/hot_air_balloon
mv $path_dataset/Sketchy/photo/tx_000000000000/jack-o-lantern $path_dataset/Sketchy/photo/tx_000000000000/jack_o_lantern
mv $path_dataset/Sketchy/extended_photo/hot-air_balloon $path_dataset/Sketchy/extended_photo/hot_air_balloon
mv $path_dataset/Sketchy/extended_photo/jack-o-lantern $path_dataset/Sketchy/extended_photo/jack_o_lantern
echo "Done"
echo "Sketchy dataset is now ready to be used"
############################################ download the TU-Berlin dataset ############################################
if [[ ! -d $path_dataset/TU-Berlin ]]; then
  mkdir $path_dataset/TU-Berlin
fi
echo "Downloading the sketches of TU-Berlin dataset (it will take some time)"
wget http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip -O $path_dataset/TU-Berlin/sketches.zip
echo -n "Unzipping it..."
unzip -qq $path_dataset/TU-Berlin/sketches.zip -d $path_dataset/TU-Berlin
rm $path_dataset/TU-Berlin/sketches.zip
mv $path_dataset/TU-Berlin/png $path_dataset/TU-Berlin/sketches
mv $path_dataset/'TU-Berlin/sketches/flying bird' $path_dataset/'TU-Berlin/sketches/flying_bird'
mv $path_dataset/'TU-Berlin/sketches/sponge bob' $path_dataset/'TU-Berlin/sketches/sponge_bob'
mv $path_dataset/'TU-Berlin/sketches/santa claus' $path_dataset/'TU-Berlin/sketches/santa_claus'
mv $path_dataset/'TU-Berlin/sketches/standing bird' $path_dataset/'TU-Berlin/sketches/standing_bird'
mv $path_dataset/'TU-Berlin/sketches/parking meter' $path_dataset/'TU-Berlin/sketches/parking_meter'
mv $path_dataset/'TU-Berlin/sketches/person walking' $path_dataset/'TU-Berlin/sketches/person_walking'
mv $path_dataset/'TU-Berlin/sketches/walkie talkie' $path_dataset/'TU-Berlin/sketches/walkie_talkie'
mv $path_dataset/'TU-Berlin/sketches/floor lamp' $path_dataset/'TU-Berlin/sketches/floor_lamp'
mv $path_dataset/'TU-Berlin/sketches/fire hydrant' $path_dataset/'TU-Berlin/sketches/fire_hydrant'
mv $path_dataset/'TU-Berlin/sketches/computer monitor' $path_dataset/'TU-Berlin/sketches/computer_monitor'
mv $path_dataset/'TU-Berlin/sketches/sea turtle' $path_dataset/'TU-Berlin/sketches/sea_turtle'
mv $path_dataset/'TU-Berlin/sketches/flower with stem' $path_dataset/'TU-Berlin/sketches/flower_with_stem'
mv $path_dataset/'TU-Berlin/sketches/bear (animal)' $path_dataset/'TU-Berlin/sketches/bear_(animal)'
mv $path_dataset/'TU-Berlin/sketches/person sitting' $path_dataset/'TU-Berlin/sketches/person_sitting'
mv $path_dataset/'TU-Berlin/sketches/bottle opener' $path_dataset/'TU-Berlin/sketches/bottle_opener'
mv $path_dataset/'TU-Berlin/sketches/race car' $path_dataset/'TU-Berlin/sketches/race_car'
mv $path_dataset/'TU-Berlin/sketches/pickup truck' $path_dataset/'TU-Berlin/sketches/pickup_truck'
mv $path_dataset/'TU-Berlin/sketches/baseball bat' $path_dataset/'TU-Berlin/sketches/baseball_bat'
mv $path_dataset/'TU-Berlin/sketches/potted plant' $path_dataset/'TU-Berlin/sketches/potted_plant'
mv $path_dataset/'TU-Berlin/sketches/power outlet' $path_dataset/'TU-Berlin/sketches/power_outlet'
mv $path_dataset/'TU-Berlin/sketches/alarm clock' $path_dataset/'TU-Berlin/sketches/alarm_clock'
mv $path_dataset/'TU-Berlin/sketches/crane (machine)' $path_dataset/'TU-Berlin/sketches/crane_(machine)'
mv $path_dataset/'TU-Berlin/sketches/traffic light' $path_dataset/'TU-Berlin/sketches/traffic_light'
mv $path_dataset/'TU-Berlin/sketches/hot air balloon' $path_dataset/'TU-Berlin/sketches/hot_air_balloon'
mv $path_dataset/'TU-Berlin/sketches/car (sedan)' $path_dataset/'TU-Berlin/sketches/car_(sedan)'
mv $path_dataset/'TU-Berlin/sketches/mouse (animal)' $path_dataset/'TU-Berlin/sketches/mouse_(animal)'
mv $path_dataset/'TU-Berlin/sketches/paper clip' $path_dataset/'TU-Berlin/sketches/paper_clip'
mv $path_dataset/'TU-Berlin/sketches/pipe (for smoking)' $path_dataset/'TU-Berlin/sketches/pipe_(for_smoking)'
mv $path_dataset/'TU-Berlin/sketches/space shuttle' $path_dataset/'TU-Berlin/sketches/space_shuttle'
mv $path_dataset/'TU-Berlin/sketches/satellite dish' $path_dataset/'TU-Berlin/sketches/satellite_dish'
mv $path_dataset/'TU-Berlin/sketches/palm tree' $path_dataset/'TU-Berlin/sketches/palm_tree'
mv $path_dataset/'TU-Berlin/sketches/flying saucer' $path_dataset/'TU-Berlin/sketches/flying_saucer'
mv $path_dataset/'TU-Berlin/sketches/cell phone' $path_dataset/'TU-Berlin/sketches/cell_phone'
mv $path_dataset/'TU-Berlin/sketches/door handle' $path_dataset/'TU-Berlin/sketches/door_handle'
mv $path_dataset/'TU-Berlin/sketches/human-skeleton' $path_dataset/'TU-Berlin/sketches/human_skeleton'
mv $path_dataset/'TU-Berlin/sketches/hot-dog' $path_dataset/'TU-Berlin/sketches/hot_dog'
mv $path_dataset/'TU-Berlin/sketches/frying-pan' $path_dataset/'TU-Berlin/sketches/frying_pan'
mv $path_dataset/'TU-Berlin/sketches/beer-mug' $path_dataset/'TU-Berlin/sketches/beer_mug'
mv $path_dataset/'TU-Berlin/sketches/speed-boat' $path_dataset/'TU-Berlin/sketches/speed_boat'
mv $path_dataset/'TU-Berlin/sketches/t-shirt' $path_dataset/'TU-Berlin/sketches/t_shirt'
mv $path_dataset/'TU-Berlin/sketches/ice-cream-cone' $path_dataset/'TU-Berlin/sketches/ice_cream_cone'
mv $path_dataset/'TU-Berlin/sketches/wine-bottle' $path_dataset/'TU-Berlin/sketches/wine_bottle'
mv $path_dataset/'TU-Berlin/sketches/computer-mouse' $path_dataset/'TU-Berlin/sketches/computer_mouse'
mv $path_dataset/'TU-Berlin/sketches/wrist-watch' $path_dataset/'TU-Berlin/sketches/wrist_watch'
mv $path_dataset/'TU-Berlin/sketches/teddy-bear' $path_dataset/'TU-Berlin/sketches/teddy_bear'
mv $path_dataset/'TU-Berlin/sketches/head-phones' $path_dataset/'TU-Berlin/sketches/head_phones'
mv $path_dataset/'TU-Berlin/sketches/tennis-racket' $path_dataset/'TU-Berlin/sketches/tennis_racket'
echo "Done"
echo "Downloading the images of TU-Berlin dataset (it will take some time)"
python3 src/download_gdrive.py 0B2U-hnwRkpRrMFVvTmFQa3dmSUk $path_dataset/TU-Berlin/images.zip
echo -n "Unzipping it..."
unzip -qq $path_dataset/TU-Berlin/images.zip -d $path_dataset/TU-Berlin
rm $path_dataset/TU-Berlin/images.zip
mv $path_dataset/TU-Berlin/ImageResized $path_dataset/TU-Berlin/images
mv $path_dataset/'TU-Berlin/images/flyingbird' $path_dataset/'TU-Berlin/images/flying_bird'
mv $path_dataset/'TU-Berlin/images/spongebob' $path_dataset/'TU-Berlin/images/sponge_bob'
mv $path_dataset/'TU-Berlin/images/santaclaus' $path_dataset/'TU-Berlin/images/santa_claus'
mv $path_dataset/'TU-Berlin/images/standingbird' $path_dataset/'TU-Berlin/images/standing_bird'
mv $path_dataset/'TU-Berlin/images/parkingmeter' $path_dataset/'TU-Berlin/images/parking_meter'
mv $path_dataset/'TU-Berlin/images/personwalking' $path_dataset/'TU-Berlin/images/person_walking'
mv $path_dataset/'TU-Berlin/images/walkietalkie' $path_dataset/'TU-Berlin/images/walkie_talkie'
mv $path_dataset/'TU-Berlin/images/floorlamp' $path_dataset/'TU-Berlin/images/floor_lamp'
mv $path_dataset/'TU-Berlin/images/firehydrant' $path_dataset/'TU-Berlin/images/fire_hydrant'
mv $path_dataset/'TU-Berlin/images/computermonitor' $path_dataset/'TU-Berlin/images/computer_monitor'
mv $path_dataset/'TU-Berlin/images/seaturtle' $path_dataset/'TU-Berlin/images/sea_turtle'
mv $path_dataset/'TU-Berlin/images/flowerwithstem' $path_dataset/'TU-Berlin/images/flower_with_stem'
mv $path_dataset/'TU-Berlin/images/bearanimal' $path_dataset/'TU-Berlin/images/bear_(animal)'
mv $path_dataset/'TU-Berlin/images/personsitting' $path_dataset/'TU-Berlin/images/person_sitting'
mv $path_dataset/'TU-Berlin/images/bottleopener' $path_dataset/'TU-Berlin/images/bottle_opener'
mv $path_dataset/'TU-Berlin/images/racecar' $path_dataset/'TU-Berlin/images/race_car'
mv $path_dataset/'TU-Berlin/images/pickuptruck' $path_dataset/'TU-Berlin/images/pickup_truck'
mv $path_dataset/'TU-Berlin/images/baseballbat' $path_dataset/'TU-Berlin/images/baseball_bat'
mv $path_dataset/'TU-Berlin/images/pottedplant' $path_dataset/'TU-Berlin/images/potted_plant'
mv $path_dataset/'TU-Berlin/images/poweroutlet' $path_dataset/'TU-Berlin/images/power_outlet'
mv $path_dataset/'TU-Berlin/images/alarmclock' $path_dataset/'TU-Berlin/images/alarm_clock'
mv $path_dataset/'TU-Berlin/images/cranemachine' $path_dataset/'TU-Berlin/images/crane_(machine)'
mv $path_dataset/'TU-Berlin/images/trafficlight' $path_dataset/'TU-Berlin/images/traffic_light'
mv $path_dataset/'TU-Berlin/images/hotairballoon' $path_dataset/'TU-Berlin/images/hot_air_balloon'
mv $path_dataset/'TU-Berlin/images/carsedan' $path_dataset/'TU-Berlin/images/car_(sedan)'
mv $path_dataset/'TU-Berlin/images/mouse' $path_dataset/'TU-Berlin/images/mouse_(animal)'
mv $path_dataset/'TU-Berlin/images/paperclip' $path_dataset/'TU-Berlin/images/paper_clip'
mv $path_dataset/'TU-Berlin/images/pipe' $path_dataset/'TU-Berlin/images/pipe_(for_smoking)'
mv $path_dataset/'TU-Berlin/images/spacesshuttle' $path_dataset/'TU-Berlin/images/space_shuttle'
mv $path_dataset/'TU-Berlin/images/satellitedish' $path_dataset/'TU-Berlin/images/satellite_dish'
mv $path_dataset/'TU-Berlin/images/palmtree' $path_dataset/'TU-Berlin/images/palm_tree'
mv $path_dataset/'TU-Berlin/images/flyingsaucer' $path_dataset/'TU-Berlin/images/flying_saucer'
mv $path_dataset/'TU-Berlin/images/cellphone' $path_dataset/'TU-Berlin/images/cell_phone'
mv $path_dataset/'TU-Berlin/images/doorhandle' $path_dataset/'TU-Berlin/images/door_handle'
mv $path_dataset/'TU-Berlin/images/humanskeleton' $path_dataset/'TU-Berlin/images/human_skeleton'
mv $path_dataset/'TU-Berlin/images/hotdog' $path_dataset/'TU-Berlin/images/hot_dog'
mv $path_dataset/'TU-Berlin/images/fryingpan' $path_dataset/'TU-Berlin/images/frying_pan'
mv $path_dataset/'TU-Berlin/images/beermug' $path_dataset/'TU-Berlin/images/beer_mug'
mv $path_dataset/'TU-Berlin/images/speedboat' $path_dataset/'TU-Berlin/images/speed_boat'
mv $path_dataset/'TU-Berlin/images/tshirt' $path_dataset/'TU-Berlin/images/t_shirt'
mv $path_dataset/'TU-Berlin/images/icecreamcone' $path_dataset/'TU-Berlin/images/ice_cream_cone'
mv $path_dataset/'TU-Berlin/images/winebottle' $path_dataset/'TU-Berlin/images/wine_bottle'
mv $path_dataset/'TU-Berlin/images/computermouse' $path_dataset/'TU-Berlin/images/computer_mouse'
mv $path_dataset/'TU-Berlin/images/wristwatching' $path_dataset/'TU-Berlin/images/wrist_watch'
mv $path_dataset/'TU-Berlin/images/teddybear' $path_dataset/'TU-Berlin/images/teddy_bear'
mv $path_dataset/'TU-Berlin/images/headphones' $path_dataset/'TU-Berlin/images/head_phones'
mv $path_dataset/'TU-Berlin/images/tennisracket' $path_dataset/'TU-Berlin/images/tennis_racket'
mv $path_dataset/'TU-Berlin/images/tromobone' $path_dataset/'TU-Berlin/images/trombone'
mv $path_dataset/'TU-Berlin/images/chandeler' $path_dataset/'TU-Berlin/images/chandelier'
mv $path_dataset/'TU-Berlin/images/griaffe' $path_dataset/'TU-Berlin/images/giraffe'
mv $path_dataset/'TU-Berlin/images/diamod' $path_dataset/'TU-Berlin/images/diamond'
mv $path_dataset/'TU-Berlin/images/spidar' $path_dataset/'TU-Berlin/images/spider'
find $path_dataset/TU-Berlin/images -type f -name '*.JPEG' -print0 | xargs -0 rename 's/\.JPEG/\.jpg/'
echo "Done"
echo "TU-Berlin dataset is now ready to be used"
