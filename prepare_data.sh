mkdir datasets
cd datasets
pip uninstall -y gdown
pip install -U --no-cache-dir gdown

mkdir crowdhuman
cd crowdhuman
gdown --id 1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3
gdown --id 10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL
gdown --id 134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y
unzip CrowdHuman_train01.zip
rm -r CrowdHuman_train01.zip
gdown --id 17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla
unzip CrowdHuman_train02.zip
rm -r CrowdHuman_train02.zip
gdown --id 1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW
unzip CrowdHuman_train03.zip
rm -r CrowdHuman_train03.zip
mv Images/ CrowdHuman_train

gdown --id 18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO
unzip CrowdHuman_val.zip
rm -r CrowdHuman_val.zip
mv Images/ CrowdHuman_val

cd ..
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
mv MOT17 mot
rm -r MOT17.zip
