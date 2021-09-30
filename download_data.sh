output_dir=$1
echo "Downloading data from HAW Cloud"
wget --user=aco910 --ask-password "https://cloud.haw-hamburg.de/remote.php/dav/files/aco910/miniaturautonomie_lanedetection/image_data.zip"
mkdir ${output_dir}/data
mv image_data.zip ${output_dir}/data/
cd ${output_dir}/data
echo "Unzipping"
unzip image_data.zip
rm image_data.zip
