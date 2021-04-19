import torch
import constants
import data
import metrics
import grumodel
from torch import nn

def train():
    device = torch.device(constants.DEVICE)

    dataset = data.Dataset()
    train_data = dataset.train_dataloader
    test_data = dataset.test_dataloader

    model = grumodel.GRUModel(input_size= constants.INPUT_SIZE, hidden_size=constants.HIDDEN_SIZE, num_classes=constants.OUT_SIZE)\
        .to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = constants.LR)
    criterion = nn.CrossEntropyLoss()

    # PLOT
    # pp = ProgressPlot(plot_names=['Accuracy', 'Loss'], line_names = ['Train', 'Test'])
    # loss_history = []

    for epoch in range(constants.EPOCH):
        for i, (x, y, c) in enumerate(train_data):
            optimizer.zero_grad()
            out = model(x.to(device))
            loss = criterion(out, y.to(device))
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                model.initHidden()
            
            # PLOT
            # loss_history.append(loss)
            #
            # if len(loss_history) == 2000:
            #     loss_sum = sum(loss_history).item()
            #     acc_data = torch.utils.data.Subset(train_data, range(0, 700))
            #     pp.update([[GetAccuracy(model, acc_data), GetAccuracy(model, test_data)], 
            #             [loss_sum, loss_sum]])
            #     loss_history=[]

    metrics.showMetrics(model, test_data, f"\nEpoch: {epoch} Test")
    metrics.showMetrics(model, dataset.neg_rand_dataset + dataset.pos_clean_dataset, f'Epoch {epoch} Clear')

    torch.save(model, constants.TRAINED_MODEL_NAME)


if __name__ == '__main__':
    train()
