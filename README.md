# image_inpainting
my daily work  

git clone https://github.com/mengwanglalala/image_inpainting.git

python train.py --model 1 --checkpoints ./checkpoints

nohup python train.py --model 1 --checkpoints ./checkpoints >> /home/wangmeng/work/image_inpainting/checkpoints/env.log 2>&1 &

使用单个2080ti运行程序时推荐参数：
edge model
线程16，pin_memory=True
BATCH_SIZE: 36               # input batch size for training



inpainting model
线程16，pin_memory=True
BATCH_SIZE: 16               # input batch size for training
