def accelerator():
    # ddp = DistributedDataParallel
    trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp')


def accumulate_grad_batches():
    # Accumulates grads every k batches or as set up in the dict. Trainer also calls optimizer.step()
    # for the last indivisible step number.

    # default used by the Trainer (no accumulation)
    trainer = Trainer(accumulate_grad_batches=1)

    # Example:

    # accumulate every 4 batches (effective batch size is batch*4)
    trainer = Trainer(accumulate_grad_batches=4)

    # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
    trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})


def auto_scale_batch_size():
    # Automatically tries to find the largest batch size that fits into memory, before any training.

    # default used by the Trainer (no scaling of batch size)
    trainer = Trainer(auto_scale_batch_size=None)

    # run batch size scaling, result overrides hparams.batch_size
    trainer = Trainer(auto_scale_batch_size='binsearch')

    # call tune to find the batch size
    trainer.tune(model)


def auto_select_gpus():
    # If enabled and gpus is an integer, pick available gpus automatically. This is especially useful
    # when GPUs are
    # configured to be in “exclusive mode”, such that only one process at a time can access them.
    # no auto selection (picks first 2 gpus on system, may fail if other process is occupying)
    trainer = Trainer(gpus=2, auto_select_gpus=False)

    # enable auto selection (will find two available gpus on system)
    trainer = Trainer(gpus=2, auto_select_gpus=True)

    # specifies all GPUs regardless of its availability
    Trainer(gpus=-1, auto_select_gpus=False)

    # specifies all available GPUs (if only one GPU is not occupied, uses one gpu)
    Trainer(gpus=-1, auto_select_gpus=True)


def auto_lr_find():
    # Runs a learning rate finder algorithm when calling trainer.tune(),
    # to find optimal initial learning rate.

    # default used by the Trainer (no learning rate finder)
    trainer = Trainer(auto_lr_find=False)

    # Example:

    # run learning rate finder, results override hparams.learning_rate
    trainer = Trainer(auto_lr_find=True)

    # call tune to find the lr
    trainer.tune(model)
    # run learning rate finder, results override hparams.my_lr_arg
    trainer = Trainer(auto_lr_find='my_lr_arg')

    # call tune to find the lr
    trainer.tune(model)


def call_backs():
    # a list of callbacks
    callbacks = [PrintCallback()]
    trainer = Trainer(callbacks=callbacks)

    # Example:

    from pytorch_lightning.callbacks import Callback

    class PrintCallback(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is started!")

        def on_train_end(self, trainer, pl_module):
            print("Training is done.")


def check_val_every_n_epoch():
    # default used by the Trainer
    trainer = Trainer(check_val_every_n_epoch=1)

    # run val loop every 10 training epochs
    trainer = Trainer(check_val_every_n_epoch=10)


def limit_train_batches():
    # How much of training dataset to check. Useful when debugging or testing something that happens
    # at the end of an epoch.

    # default used by the Trainer
    trainer = Trainer(limit_train_batches=1.0)

    # default used by the Trainer
    trainer = Trainer(limit_train_batches=1.0)

    # run through only 25% of the training set each epoch
    trainer = Trainer(limit_train_batches=0.25)

    # run through only 10 batches of the training set each epoch
    trainer = Trainer(limit_train_batches=10)
    # it also has limit_test_batches and limit_val_batches


def log_gpu_memory():
    # default used by the Trainer
    trainer = Trainer(log_gpu_memory=None)

    # log all the GPUs (on master node only)
    trainer = Trainer(log_gpu_memory='all')

    # log only the min and max memory on the master node
    trainer = Trainer(log_gpu_memory='min_max')


def logger():
    # Logger (or iterable collection of loggers) for experiment tracking.
    # A True value uses the default TensorBoardLogger shown below. False will disable logging.

    from pytorch_lightning.loggers import TensorBoardLogger

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )
    Trainer(logger=logger)


def num_nodes():
    # Number of GPU nodes for distributed training.

    # default used by the Trainer
    trainer = Trainer(num_nodes=1)

    # to train on 8 nodes
    trainer = Trainer(num_nodes=8)


def num_sanity_val_steps():
    # default used by the Trainer
    trainer = Trainer(num_sanity_val_steps=2)

    # turn it off
    trainer = Trainer(num_sanity_val_steps=0)

    # check all validation data
    trainer = Trainer(num_sanity_val_steps=-1)


def overfit_batches():
    # Uses this much data of the training set. If nonzero, will use the same training set for
    # validation and testing. If the training dataloaders have shuffle=True, Lightning will
    # automatically disable it.

    # Useful for quickly debugging or trying to overfit on purpose.

    # default used by the Trainer
    trainer = Trainer(overfit_batches=0.0)

    # use only 1% of the train set (and use the train set for val and test)
    trainer = Trainer(overfit_batches=0.01)

    # overfit on 10 of the same batches
    trainer = Trainer(overfit_batches=10)


def truncated_bptt_steps():
    # Truncated back prop breaks performs backprop every k steps of a much longer sequence.

    # If this is enabled, your batches will automatically get truncated and the trainer will apply
    # Truncated Backprop to it.

    # (Williams et al. “An efficient gradient-based algorithm for on-line training of recurrent network trajectories.”)

    # default used by the Trainer (ie: disabled)
    trainer = Trainer(truncated_bptt_steps=None)

    # backprop every 5 steps in a batch
    trainer = Trainer(truncated_bptt_steps=5)
