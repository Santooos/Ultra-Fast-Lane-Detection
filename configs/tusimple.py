# DATA
dataset='Tusimple'
data_root = r"C:\Users\sant4\Desktop\TUSimple\test_set"

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = r"E:\Senior Year Project\TestRes"

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = r"E:\Senior Year Project\LogPathTUSimple\20240410_103141_lr_4e-04_b_32\ep099.pth"
test_work_dir = log_path = r"E:\Senior Year Project\TestResTUSimple"

num_lanes = 4
cls_num_per_lane = num_lanes  # Directly use your existing 'num_lanes' variable
