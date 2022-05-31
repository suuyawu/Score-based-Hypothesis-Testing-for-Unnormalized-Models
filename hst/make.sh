python make.py --mode ptb --data MVN --run test --round 16
python make.py --mode ptb --data GMM --run test --round 16
python make.py --mode ptb --data RBM --run test --round 16
python make.py --mode ptb --data EXP --run test --round 16

python make.py --mode ds --data MVN --run test --round 16
python make.py --mode ds --data GMM --run test --round 16 --split_round 6
python make.py --mode ds --data RBM --run test --round 16
python make.py --mode ds --data EXP --run test --round 16


