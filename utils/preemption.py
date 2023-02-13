import os
import signal
import torch


CHECKPOINT_NAME = 'job_suspension_checkpoint.pth'


def graceful_exit_handler(signum, epoch, net, optimizer):
    print('Job suspended ({}) on epoch {}'.format(str(signum), epoch))
    # save your checkpoints to shared storage
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_NAME)

    # exit with status "1" is important for the Job to return later.
    exit(1)


def resume_run(net, optim):
    checkpoint = torch.load(CHECKPOINT_NAME)
    net.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    id = os.environ['WANDB_ID']
    return net, optim, epoch, id
