from training import Trainer
from helpers import log_mets, get_minutes_left, epochs_loop, log_message
from data.loaders import get_data_loaders

def run(config, state=None):
    trainer = Trainer(model, loss_fn, optimizer, metrics, state=state)

    train_loader, train_eval_loader, val_loader = get_data_loaders(config)

    initial_epoch = trainer.epoch
    for epoch in epochs_loop(trainer.epoch, config.epochs, get_minutes_left() - 10):
        try:
            log_message(f"\n== epoch {trainer.epoch} ==")
            trainer.train(train_loader)
            trainer.test(train_eval_loader)
            log_mets(trainer.metrics.compute(), epoch, "train")
            trainer.test(val_loader)
            log_mets(trainer.metrics.compute(), epoch, "val", True)
            trainer.epoch += 1
        except Exception as e:
            if epoch == initial_epoch:
                raise e
            print(f'encountered: {e}')
            print('trying to exit script and save checkpoint...')
            break

    return trainer
