from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):
    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        torch.manual_seed(epoch)
        total_accuracy = []
        for iteration in range(20):
            dummy_training_loss = 0.9**(epoch+iteration/20.)
            dummy_training_accuracy = epoch/10. + torch.randn(10)
            train_logger.add_scalar('loss', dummy_training_loss, global_step=global_step)
            total_accuracy.append(dummy_training_accuracy)
            global_step += 1
        train_logger.add_scalar('accuracy', torch.mean(torch.cat(total_accuracy)), global_step=global_step)

        total_accuracy = []
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            total_accuracy.append(dummy_validation_accuracy)
        valid_logger.add_scalar('accuracy', torch.mean(torch.cat(total_accuracy)), global_step=global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
