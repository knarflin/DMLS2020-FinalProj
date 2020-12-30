from argparser import get_args
from model import Classifier
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch, os

def statistic(DataPath):
	
	transform = transforms.Compose([
		transforms.Resize((args['size'], args['size'])), 
		transforms.ToTensor(), 
	])
	dataset = ImageFolder(DataPath, transform = transform)

	count = torch.IntTensor([0] * 43)
	mean = torch.stack([image.mean((1, 2)) for image, _ in dataset]).mean(0)
	std = torch.stack([image for image, _ in dataset]).std((0, 2, 3))
	for label in range(43):
		count[label] = len(os.listdir(DataPath + '%05d/'%label))

	del transform, dataset

	return tuple(mean), tuple(std), count

def forward(DataLoader, model, LossFunction, optimizer = None) :

	TotalLoss = 0.0
	count = 0
	correct = 0

	for inputs, labels in DataLoader:
		# initialize
		torch.cuda.empty_cache()
		if optimizer :
			optimizer.zero_grad()
			
		# forward
		inputs = inputs.cuda()
		outputs = model(inputs)
		del inputs

		# loss and step
		labels = labels.cuda()
		loss = LossFunction(outputs, labels)
		TotalLoss += loss.item()

		if optimizer :
			loss.backward()
		del loss
		if optimizer :
			optimizer.step()
		
		# convert to prediction
		tmp, pred = outputs.max(1)
		del tmp, outputs

		correct += (pred == labels).sum().item()
		count += len(labels)
		del pred, labels

	print('%.3f'%(TotalLoss / len(DataLoader)), '%.2f%%'%(correct / count * 100))
	return correct / count

if __name__ == '__main__':

	args = get_args()
	ModelPath = '../../'
	DataPath = '../../splitdata/'

	mean, std, count = statistic(DataPath + 'train/site%d/'%args['site'])

	transform = transforms.Compose([
		transforms.Resize((args['size'], args['size'])), 
		transforms.ToTensor(), 
		transforms.Normalize(mean, std)
	])

	total = sum(count)
	weight = []
	for number in count:
		weight += [total / number] * number

	sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor(weight), int(total))
	del count, weight

	TrainingSet = ImageFolder(root = DataPath + 'train/site%d/'%args['site'], transform = transform)
	ValidationSet = ImageFolder(root = DataPath + 'validation/', transform = transform)

	TrainingLoader = DataLoader(TrainingSet, batch_size = args['bs'], num_workers = os.cpu_count() // 2, pin_memory = True, drop_last = True, sampler = sampler, prefetch_factor = 1)
	ValidationLoader = DataLoader(ValidationSet, batch_size = args['bs'], num_workers = os.cpu_count() // 2, prefetch_factor = 1)

	if args['load']:
		model = torch.load(ModelPath + args['load'])
	else:
		model = Classifier().cuda()

	torch.backends.cudnn.benchmark = True
	LossFunction = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])

	model.eval()
	with torch.no_grad() :
		print('Validatoin', end = ' ')
		BestAccuracy = forward(ValidationLoader, model, LossFunction)

	if args['save']:
		torch.save(model, ModelPath + args['save'])

	for epoch in range(args['ep']) :
		print('\nEpoch : %2d'%epoch)

		model.train()
		print('Training', end = ' ')
		forward(TrainingLoader, model, LossFunction, optimizer)

		model.eval()
		with torch.no_grad() :
			print('Validatoin', end = ' ')
			accuracy = forward(ValidationLoader, model, LossFunction)

		if args['save'] and accuracy > BestAccuracy:
			BestAccuracy = accuracy
			torch.save(model, ModelPath + args['save'])