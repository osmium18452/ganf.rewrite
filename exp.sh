gpu=3
#python main.py -GANT -D save/22.3.23/swat/1 -g $gpu -d swat -s 1  -b 1000
#python main.py -GANT -D save/22.3.23/swat/2 -g $gpu -d swat -s 10
#python main.py -GAN  -D save/22.3.23/swat/3 -g $gpu -d swat -s 1  -b 1000
#python main.py -GAN  -D save/22.3.23/swat/4 -g $gpu -d swat -s 10

python main.py -GANT -D save/22.3.23/wadi.old/1 -g $gpu -d wadi.old -s 1  -b 500
python main.py -GANT -D save/22.3.23/wadi.old/2 -g $gpu -d wadi.old -s 10
python main.py -GAN  -D save/22.3.23/wadi.old/3 -g $gpu -d wadi.old -s 1  -b 500
#python main.py -GAN  -D save/22.3.23/wadi.old/4 -g $gpu -d wadi.old -s 10

python main.py -GANT -D save/22.3.23/wadi.new/1 -g $gpu -d wadi.new -s 1  -b 500
#python main.py -GANT -D save/22.3.23/wadi.new/2 -g $gpu -d wadi.new -s 10
python main.py -GAN  -D save/22.3.23/wadi.new/3 -g $gpu -d wadi.new -s 1  -b 500
#python main.py -GAN  -D save/22.3.23/wadi.new/4 -g $gpu -d wadi.new -s 10

