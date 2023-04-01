import torch.nn as nn
import backbones
from PI_layers import PhiLayer_1D

class PINet(nn.Module):
    def __init__(self, num_class,
                 sim_feature,
                 base_net='alexnet',
                 use_bottleneck=False,
                 use_PIlayer = True,
                 bottleneck_width=256,
                 max_iter=1000, **kwargs):
        super(PINet, self).__init__()
        self.pi_layer = PhiLayer_1D(sim_feature, num_class)
        self.base_network = backbones.get_backbone(base_net)
        self.use_PIlayer = use_PIlayer
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        if feature_dim != num_class:
            classifier_list = [
                # nn.Linear(feature_dim, num_class),
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_class)
            ]
            self.classifier_layer = nn.Sequential(*classifier_list)

        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, source, source_label):
        if self.use_PIlayer:
            source = self.pi_layer(source)
        source = self.base_network(source)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)

        return clf_loss, source_clf

    def get_parameters(self, initial_lr=1.0):
        # params = [
        #     {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
        #     {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        # ]
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )

        return params


    def predict(self, x):
        x = self.base_network(x)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        clf = self.classifier_layer(x)
        return clf

    def output_feature(self, x):
        feature = self.base_network(x)
        return feature