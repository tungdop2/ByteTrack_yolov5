mkdir datasets
cd datasets

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

gdown --id 19QyGOCqn8K_rc9TXJ8UwLSxCx17e0GoY
unzip ETHZ.zip
rm -r ETHZ.zip

gdown --id 1DgLHqEkQUOj63mCrS_0UGFEM9BG8sIZs
gdown --id 1BH9Xz59UImIGUdYwUR-cnP1g7Ton_LcZ
gdown --id 1q_OltirP68YFvRWgYkBHLEFSUayjkKYE
gdown --id 1VSL0SFoQxPXnIdBamOZJzHrHJ1N2gsTW
zip -s- Citypersons.zip -O combine.zip
unzip combine.zip
rm -r combine.zip
mv Citypersons Cityscapes
rm -r Citypersons.zip
rm -r Citypersons.z01
rm -r Citypersons.z02
rm -r Citypersons.z03

wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
mv MOT17 mot
rm -r MOT17.zip
