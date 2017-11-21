import torch
import torch.nn
from torch.nn.modules.loss import _Loss


class CenterLoss(_Loss):
    """ Implements the CenterLoss as proposed in 'A Discriminative Feature Learning Approach
        for Deep Face Recognition' (https://ydwen.github.io/papers/WenECCV16.pdf).
        The centers are updated manually.
        The forward method behaves differently depending on the training state of the model:
        it only updates the center during training.
    """
    def __init__(self, model, centers, num_classes, alpha=0.5):
        """ Constructor.
            Parameters:
                - model: the model whose weights are being optimized. It is needed to check
                    the training state.
                - centers: FloatTensor. Shape (num_classes, features_dim).
                - num_classes: number of classes for classification.
                - alpha: update rate of the centers.
        """
        super(CenterLoss, self).__init__(size_average=False)
        self.model = model
        self.centers = centers
        assert isinstance(alpha, float) and 0 < alpha <= 1, "'alpha' must be a float between 0 and 1."
        self.alpha = alpha
        assert num_classes == self.centers.size(0), "'centers' must have 'num_classes' rows."
        self.num_classes = num_classes
        self.features_dim = self.centers.size(1)
        

    def forward(self, features, targets):
        torch.nn.modules.loss._assert_no_grad(targets)
        
        batch_size = features.size(0)
        centers_var = Variable(self.centers)
        
        # Builds a tensor of shape (batch_size, features_dim), where the i-th row is the 
        # corresponding center of the sample i (class label y_i)
        batch_centers = torch.index_select(centers_var, dim=0, index=targets.data)
    
        # Difference between the sample features and their respective centers
        diff = features - batch_centers
        
        # Calculates the average center loss
        center_loss = 0.5 * torch.sum(torch.pow(diff, 2)) / batch_size

        # Updates the centers in the training stage
        if self.model.training:
            # Each center Cj is attracted by the corresponding features in the batch. Each center
            # update is divided by (Nj+1), where Nj is the number of occurences of the class j in
            # the current batch.
            
            # torch.histc must be calculated in the CPU (no cuda implementation)
            if targets.is_cuda:
                hist = (torch.histc(targets.data.cpu().float(), self.num_classes, 
                                min=0, max=self.num_classes-1) + 1).cuda()
            else:
                hist = (torch.histc(targets.data.float(), self.num_classes, 
                                min=0, max=self.num_classes-1) + 1)
            class_occurences = torch.index_select(hist, dim=0, index=targets.data).view(-1, 1)
            
            # Center updates
            center_updates = self.alpha * torch.div(diff.data, denom)
            
            # Updates the centers in-place: the variable `targets` indexes the centers that must be
            # updated. `targets` is expanded to have the same shape as `center_updates`.
            self.centers.scatter_add_(0, targets.data.long().view(-1,1).expand(batch_size,self.features_dim),
                                      center_updates)
      
        return center_loss