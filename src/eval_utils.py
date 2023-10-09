import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_dataset, split_dataset
from models import ResNet18
from client_utils import DatasetSplit, get_client_labels


def grad_cos_sim(grads_i, grads_j):
    def grad_flatten(grads):
        all_grads = []
        for k, _ in grads.items():
            all_grads.append(grads[k].cpu().ravel())
        return np.hstack(all_grads).ravel()

    grads_i = grad_flatten(grads_i)
    grads_j = grad_flatten(grads_j)

    return np.dot(grads_i, grads_j) / (np.linalg.norm(grads_i) * np.linalg.norm(grads_j))

def grad_diversity(grad_arr):
    def grad_flatten(grads):
        all_grads = []
        for k, _ in grads.items():
            all_grads.append(grads[k].cpu().ravel())
        return np.hstack(all_grads).ravel()

    flattened = []
    #flatten all grads
    for g in grad_arr:
        flattened.append(grad_flatten(g[0]))

    num = np.linalg.norm(flattened[0]) ** 2
    denom = np.dot(flattened[0], flattened[0])
    for i in range(1, len(flattened)):
        num += np.linalg.norm(flattened[i]) ** 2
        denom += np.dot(flattened[i], flattened[1])
    denom = np.linalg.norm(denom)

    return num / denom

def grad_diff(grads_i, grads_j):
    def grad_flatten(grads):
        all_grads = []
        for k, _ in grads.items():
            all_grads.append(grads[k].cpu().ravel())
        return np.hstack(all_grads).ravel()

    grads_i = grad_flatten(grads_i)
    grads_j = grad_flatten(grads_j)
    return np.linalg.norm(grads_j - grads_i)

def validation_inference(args, model, validation_dataset, num_workers):
    """
    Returns the validation accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    valloader = DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    for batch_idx, (images, labels) in enumerate(valloader):
        images, labels = images.to(args.device), labels.to(args.device)

        # Inference
        try:
            outputs = model(images)
        except:
            outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss / (batch_idx + 1)
    return accuracy, loss

def celeba_inference(model, dataset, device, num_workers):
    """
    Returns the validation accuracy and loss.
    """
    model.eval()
    loss, total = 0.0, 0.0
    correct_per_feature = [0 for _ in range(40)]
    criterion = torch.nn.BCELoss().to(device)
    valloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)

    for batch_idx, (images, labels) in enumerate(valloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        cor = torch.eq(outputs, labels).int()
        correct_per_feature += cor
        total += len(labels)

    correct_per_feature /= 40
    print(f'correct_per_feature {correct_per_feature}')
    correct = torch.sum(correct_per_feature).item()
    print(f'correct {correct}')

    accuracy = correct/total
    loss = loss / (batch_idx + 1)
    return accuracy, correct_per_feature, loss

def test_inference(args, model, test_dataset, num_workers):
    """
    Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(args.device), labels.to(args.device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        '''print(f'pred_labels {pred_labels}')
        print(f'labels {labels}')'''

    accuracy = correct/total
    return accuracy, loss

def test_model(args, model_path, wsm=False):
    #args.dataset = 'femnist'
    model = ResNet18(args=args)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    _, _, test_dataset = get_dataset(args)
    if wsm:
        user_groups = split_dataset(test_dataset, args)
        wsm_acc, _ = wsm_inference(args, model, test_dataset, user_groups, args.num_workers)
        test_acc, _ = test_inference(args, model, test_dataset, args.num_workers)
        return test_acc, wsm_acc
    else:
        test_acc, _ = test_inference(args, model, test_dataset, args.num_workers)
        return test_acc

def wsm_inference(args, model, dataset, user_groups, num_workers):
    """
    Returns the wsm accuracy for each clients validation set using wsm.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.NLLLoss().to(args.device)
    client_proportions = get_client_labels(dataset, user_groups, num_workers, 10, proportions=True)

    for idx in range(100):
        #need both sets combined since this is entire test set
        validation_idxs = user_groups[idx]['validation']
        test_idxs = user_groups[idx]['train']
        all_idxs = np.concatenate((validation_idxs, test_idxs))
        client_dataset = DatasetSplit(dataset, all_idxs)

        testloader = DataLoader(client_dataset, batch_size=50, shuffle=False, num_workers=num_workers, pin_memory=True)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(args.device), labels.to(args.device)

            # Inference
            try:
                outputs = model(images)
                alphas = torch.from_numpy(client_proportions[idx]).to(args.device)
                log_alphas = alphas.log().clamp_(min=-1e9)
                deno = torch.logsumexp(log_alphas + outputs, dim=-1, keepdim=True)
                outputs = log_alphas + outputs - deno
                #outputs = weighted_log_softmax(outputs, client_proportions[idx])
            except:
                outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            #print(f'pred_labels {pred_labels}')
            #print(f'labels {labels}')
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    loss = loss / 100
    return accuracy, loss
