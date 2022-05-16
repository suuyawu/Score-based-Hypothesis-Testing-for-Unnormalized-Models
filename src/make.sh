python make.py --mode ptb --data MVN --run test --round 16
python make.py --mode ptb --data GMM --run test --round 16
python make.py --mode ptb --data RBM --run test --round 16

python make.py --mode ds --data MVN --run test --round 16
python make.py --mode ds --data GMM --run test --round 16
python make.py --mode ds --data RBM --run test --round 16

python make.py --mode noise --data MVN --run test --round 16
python make.py --mode noise --data GMM --run test --round 16
python make.py --mode noise --data RBM --run test --round 16

