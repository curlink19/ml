import torch
from yolo import YOLOV5m
import config
from utils.utils import try_all_gpus, BEAUTIFUL_STRING, multi_scale
from torch.optim import Adam
from load import get_dict_with_weights
from loss import YOLO_LOSS
from tqdm import tqdm


# define train_loop
def train_loop(
    model,
    loader,
    optim,
    loss_fn,
    scaler,
    epoch,
    num_epochs,
    multi_scale_training=True,
):
    print(f"Training epoch {epoch}/{num_epochs}")
    # these first 4 rows are copied from Ultralytics repo. They studied a
    # mechanism to make model's learning
    # batch_invariant for bs between 1 and 64: based on batch_size the loss is
    # cumulated in the scaler (optimizer) but the
    # frequency of scaler.step() (optim.step()) depends on the batch_size
    # check here: https://github.com/ultralytics/yolov5/issues/2377
    nbs = 64  # nominal batch size
    batch_size = len(next(iter(loader))[0])
    accumulate = max(
        round(nbs / batch_size), 1
    )  # accumulate loss before optimizing
    last_opt_step = -1

    loop = tqdm(loader)
    avg_batches_loss = 0
    loss_epoch = 0
    nb = len(loader)
    optim.zero_grad()
    for idx, (images, bboxes) in enumerate(loop):
        images = images.float() / 255
        if multi_scale_training:
            images = multi_scale(images, target_shape=640, max_stride=32)

        images = images.to(config.DEVICE, non_blocking=True)
        # BBOXES AND CLASSES ARE PUSHED to.(DEVICE) INSIDE THE LOSS_FN

        # If I had a V100...
        with torch.cuda.amp.autocast():
            out = model(images)
            loss = loss_fn(
                out,
                bboxes,
                pred_size=images.shape[2:4],
                batch_idx=idx,
                epoch=epoch,
            )
            avg_batches_loss += loss
            loss_epoch += loss

        # backpropagation
        # check docs here https://pytorch.org/docs/stable/amp.html
        scaler.scale(loss).backward()

        if idx - last_opt_step >= accumulate or (idx == nb - 1):
            scaler.unscale_(optim)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0
            )  # clip gradients
            scaler.step(optim)  # optimizer.step
            scaler.update()
            optim.zero_grad(set_to_none=True)
            last_opt_step = idx

        # update tqdm loop
        freq = 10
        if idx % freq == 0:
            loop.set_postfix(
                average_loss_batches=avg_batches_loss.item() / freq
            )
            avg_batches_loss = 0

    print(f"==> training_loss: {(loss_epoch.item() / nb):.2f}")


if __name__ == "__main__":
    devices = try_all_gpus()
    print("devices: " + str(devices) + BEAUTIFUL_STRING)

    config.DEVICE = devices[0]

    model = YOLOV5m(
        first_out=config.FIRST_OUT,
        nc=config.NC,
        anchors=config.ANCHORS,
        ch=(config.FIRST_OUT * 4, config.FIRST_OUT * 8, config.FIRST_OUT * 16),
        inference=False,
    ).to(devices[0])

    optim = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    print("trying to load weights:")
    model.load_state_dict(get_dict_with_weights())
    print("weights loaded" + BEAUTIFUL_STRING)

    loss_fn = YOLO_LOSS(model)
