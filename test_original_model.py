import torch
from torch.utils.data import DataLoader
import ordinal_entropy
from data.get_vKITTI import VKITTI
from model.model_loader import load_model
from data.transforms import ToTensor
from torchvision import transforms as T
import torch.optim as optim
from data.data_processing import unpack_and_move, inverse_depth_norm
from config_model_TTA import configure_model, collect_params
import time
from metrics import AverageMeter, Result



transformation = T.ToTensor()
trans = T.Compose([T.ToTensor()])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80
My_to_tensor = ToTensor(test=True, maxDepth=maxDepth)

# Load pre-trained model
model_original = load_model('GuideDepth', './model/weights/KITTI_Half_GuideDepth.pth')
model_original.eval().cuda()

# Load model parameter to be fine-tuned during test phase
# model = configure_model(model_original)
# params, param_names = collect_params(model)


# Prepare test dataloader for TTA
testset = VKITTI('./vkitti_testset_test/test', (192, 640))
# testset = VKITTI('/HOMES/yigao/Downloads/eval_testset/NYU_Testset', 'full')
testset_loader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)       # drop_last=True


# Define loss function and optimizer for fine-tuning
# optimizer = optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0)
# optimizer = optim.SGD(params, lr=0.001, weight_decay=0.0)
# oering the given model to make it adaptive for test data
# adapted_model = ordinal_entropy.Ordinal_entropy(model, optimizer)
# adapted_model.cuda()

# print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))

average_meter = AverageMeter()
for i, data in enumerate(testset_loader):
    t0 = time.time()
    images, gts = data
    for b in range(images.shape[0]):
        # print(b)
        packed_data = {'image': images[b], 'depth': gts[b]}
        data = My_to_tensor(packed_data)
        image, gt = unpack_and_move(data)
        # image, gt = data['image'], data['depth']
        image = image.unsqueeze(0)
        gt = gt.unsqueeze(0)

        if b >= 1:
            batched_images = torch.cat((batched_images, batched_image))
            batched_gts = torch.cat((batched_gts, batched_gt))
        else:
            batched_images = image
            batched_gts = gt
        batched_image = image
        batched_gt = gt

        batched_image.detach().cpu()
        batched_gt.detach().cpu()
        image = image.detach().cpu()
        gt = gt.detach().cpu()
    # print(batched_images.shape)
    data_time = time.time() - t0
    t0 = time.time()
    # print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
    torch.cuda.empty_cache()  # Releases all unoccupied cached memory currently held by the caching allocator
    inv_prediction, _ = model_original(batched_images)
    # inv_prediction = model_original(batched_images).detach().cpu()
    predictions = inverse_depth_norm(inv_prediction)

    batched_images = batched_images.detach().cpu()
    batched_gts = batched_gts.detach().cpu()
    predictions = predictions.detach().cpu()

    gpu_time = time.time() - t0
    # update eval metrics after each batch
    result = Result()
    result.evaluate(predictions.data, batched_gts.data)
    average_meter.update(result, gpu_time, data_time, image.size(0))

# Report eval metrics
avg = average_meter.average()
current_time = time.strftime('%H:%M', time.localtime())

print('\n*\n'
      'RMSE={average.rmse:.3f}\n'
      'MAE={average.mae:.3f}\n'
      'Delta1={average.delta1:.3f}\n'
      'Delta2={average.delta2:.3f}\n'
      'Delta3={average.delta3:.3f}\n'
      'REL={average.absrel:.3f}\n'
      'Lg10={average.lg10:.3f}\n'
      't_GPU={time:.3f}\n'.format(
    average=avg, time=avg.gpu_time))
