name = "expr"
pretrained_model = -1

train = dict(
    log_step = 10,
    model_save_step = 10000,
    num_steps = 100000,

    dataloader = dict(
        batch_size = 256,
        num_workers = 8,
        shuffle = True,
        drop_last = True
    ),

    opt = dict(
        type = "Adam",
        lr = 0.0001,
        weight_decay = 0.0001
    ),

    scheduler = dict(
        type = "MultiStepLR",
        milestones = [70000, 85000, 95000],
        gamma = 0.1
    )
)





