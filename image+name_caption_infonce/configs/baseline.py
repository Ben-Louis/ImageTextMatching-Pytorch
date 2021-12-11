_base_ = ["../common/configs/basic.py"]

name = "{{ fileBasenameNoExtension }}"
dataset = dict(use_image=True)
model = dict(swin_frozen_stage=0)
loss = dict(type="arc_infonce", scale=32, margin=0.5)

train = dict(
    dataloader = dict(batch_size=512, num_workers=16),
    opt = dict(lr=2e-5, weight_decay=1e-4),
    log_step = 5,
    num_steps = 120000,
    model_save_step = 10000,
    scheduler=dict(
        type="MultiStepLR",
        milestones=[80000, 105000],
        gamma=0.1
    ),
    grad_accum_step = 32
)

test = dict(
    dataloader = dict(
        batch_size = 128,
        num_workers = 8,
        shuffle = False,
        drop_last = False
    ),
    topk = 5
)