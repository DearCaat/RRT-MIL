import torch.optim.lr_scheduler as lr_scheduler


def define_scheduler(args, optimizer):
    if args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.num_epoch / 2, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=0)
    elif args.scheduler == 'None':
        scheduler = None
    else:
        return NotImplementedError('Scheduler [{}] is not implemented'.format(args.scheduler))
    return scheduler
