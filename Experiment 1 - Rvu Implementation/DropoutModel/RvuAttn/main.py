import time
import sys
import os 
from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPO

if __name__ == "__main__":
    game = "Pong"
    ENV = game+"NoFrameskip-v4"
    max_epochs = 60
    train_seed = 69
    GPU = "cuda:0"

    clip = 0.2
    lr = 3e-4

    n_envs = 64
    n_steps = 512

    attn_type = "RvuAttn"
    adaptive = False

    gamma = 0.99
    batch_size = 64
    v_loss_coef = 0.5

    max_grad_norm = 0.5

    xstr = 'Adaptive' if adaptive else ''
    save_dir = "runs/seed"+str(train_seed)+game+xstr+attn_type+"Dropout/" + str(time.time())

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(save_dir+"/log.txt", "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

    writer = SummaryWriter(save_dir)
    try:
 
        sys.stdout = Logger()

        ppo = PPO(
            ENV,
            max_epochs,
            n_envs,
            n_steps,
            batch_size,
            writer,
            lr=lr,
            v_loss_coef=v_loss_coef,
            max_grad_norm=max_grad_norm,
            epsilon=clip,
            train_seed = train_seed,
            cuda = GPU,
            attn_type = attn_type,
            adaptive = adaptive,
        )
        start_time = time.time()
        ppo.train()

        print("---Execution Time = %.2f seconds---" % (time.time() - start_time))

        sys.stdout.close()
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
