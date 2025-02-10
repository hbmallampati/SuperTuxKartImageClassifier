from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    import torch

    device = torch.device('cpu')

    model = CNNClassifier().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    training_data = load_data('data/train')
    testing_data = load_data('data/valid')

    global_step = 0
    for epoch in range(args.number_of_epochs):
        model.train()
        accuracy_values = []
        for img, label in training_data:
            img, label = img.to(device), label.to(device)

            temp_state = model(img)
            loss_value = loss(temp_state, label)
            accuracy_value = accuracy(temp_state, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_value, global_step)
            accuracy_values.append(accuracy_value.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            global_step += 1
        average_accuracy = sum(accuracy_values) / len(accuracy_values)

        if train_logger:
            train_logger.add_scalar('accuracy', average_accuracy, global_step)

        model.eval()
        accuracy_values = []
        for img, label in testing_data:
            img, label = img.to(device), label.to(device)
            accuracy_values.append(accuracy(model(img), label).detach().cpu().numpy())
        average_test_accuracy = sum(accuracy_values) / len(accuracy_values)

        if valid_logger:
            valid_logger.add_scalar('accuracy', average_test_accuracy, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t accuracy = %0.3f \t test accuracy = %0.3f' % (epoch, average_accuracy, average_test_accuracy))
        save_model(model)
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--number_of_epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
