import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import confusion_matrix
from pcm import plot_confusion_matrix

def val_zsl(test_att, test_X, test_label, unseen_classes, in_package):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label_zsl = np.array([])
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)

            input1 = input.reshape((input.shape[0], 1, input.shape[1])).repeat([1, test_att.shape[0], 1])
            input_linear = model.linear_x2a(input).reshape((input.shape[0], 1, test_att.shape[1])).repeat([1, test_att.shape[0], 1])

            test_att1 = test_att.reshape((1, test_att.shape[0], test_att.shape[1])).repeat([input.shape[0], 1, 1])
            test_att_linear = model.linear_a2x(test_att).reshape((1, test_att.shape[0], input.shape[1])).repeat([input.shape[0], 1, 1])

            # output = torch.norm(input1 - test_att_linear, dim=-1).pow(2) + model.alpha * torch.norm(test_att1 - input_linear, dim=-1).pow(2)
            output = torch.pow(input1 - test_att_linear, 2).sum(dim=-1) + model.alpha * torch.pow(test_att1 - input_linear, 2).sum(dim=-1)

            predicted_label_zsl = np.append(predicted_label_zsl, test_label[torch.argmin(output, 1)])

            start = end

        acc_zsl = compute_per_class_acc(test_label, predicted_label_zsl, unseen_classes)
        return acc_zsl




def val_zsl_cos(test_att, test_x, test_label, unseen_classes, in_package):
    model = in_package['model']
    # sim = cosine_similarity(test_x, torch.mm(test_att, model.W_a2x))

    # 映射到视觉空间
    sim = cosine_similarity(test_x.detach().cpu().numpy(), test_att.detach().cpu().numpy())
    # sim = cosine_similarity(test_x.detach().cpu().numpy(), model.linear_a2x(test_att).detach().cpu().numpy())
    # sim = cosine_similarity(model.linear_x2a(test_x).detach().cpu().numpy(), test_att.detach().cpu().numpy())

    # sim = (100*test_x @ test_att.T).softmax(dim=-1)
    # sim = cosine_similarity(test_x, model.linear_b2x(test_att))
    # np.save('tsne_Places_CLIP', test_x.detach().cpu().numpy())

    # 映射到语义空间
    # sim = cosine_similarity(model.linear_x2a(test_x).detach().cpu().numpy(), test_att.detach().cpu().numpy())
    # sim = cosine_similarity(model.linear_x2b(test_x), test_att)
    # np.save('tsne_MIT', model.linear_x2a(test_x).detach().cpu().numpy())
    # np.save('test_label_Places', test_label)

    # sim = torch.from_numpy(sim).float()
    # predicted_label = test_label[torch.argmax(sim, 1)]
    predicted_label = unseen_classes[np.argmax(sim, axis=1)]
    # predicted_label = unseen_classes[np.argmax(similarity.cpu().numpy(), axis=1)]
    acc = compute_per_class_acc(test_label, predicted_label, unseen_classes)
    return acc


def compute_per_class_acc(test_label, predicted_label, unseen_classes):
    acc_per_class = torch.FloatTensor(len(unseen_classes)).fill_(0)
    i = 0
    for unseen in unseen_classes:
        idx = (test_label == unseen)
        # acc_per_class[i] = np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx)
        acc_per_class[i] = np.sum(test_label[idx] == predicted_label[idx])
        i += 1
    # np.save('acc_per_class_MIT1_CLIP', acc_per_class)
    # np.save('unseen_classes_MIT', unseen_classes)

    acc = acc_per_class.sum() / test_label.shape[0]
    # if acc >= 0.974:
    #     cm = confusion_matrix(test_label, predicted_label)
    #     plot_confusion_matrix(cm, unseen_classes, "Confusion Matrix", is_norm=True)
    return acc.item()


def eval_zsl(test_a, test_x, test_label, unseenclasses, model, device, batch_size=50):
    in_package = {'model': model, 'device': device, 'batch_size': batch_size}
    model.eval()
    with torch.no_grad():
        # acc_zsl = val_zsl(test_a, test_x, test_label, unseenclasses, in_package)
        acc_zsl = val_zsl_cos(test_a, test_x, test_label, unseenclasses, in_package)
    return acc_zsl
