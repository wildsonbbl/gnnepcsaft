curl -O -L https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
curl -O -L https://github.com/zmeri/PC-SAFT/archive/refs/tags/v1.5.0.zip
unzip v1.5.0.zip
unzip eigen-master.zip -d PC-SAFT-1.5.0/externals
mv -v PC-SAFT-1.5.0/externals/eigen-master PC-SAFT-1.5.0/externals/eigen
sed -i "s/np.float_/np.float64/g" PC-SAFT-1.5.0/pcsaft.pyx
pip install ./PC-SAFT-1.5.0
