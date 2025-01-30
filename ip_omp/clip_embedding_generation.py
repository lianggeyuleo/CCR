import os

# these lines are needed to prevent pytorch from using all CPU cores
os.system("export MKL_NUM_THREADS=4")
os.system("export NUMEXPR_NUM_THREADS=4")
os.system("export OMP_NUM_THREADS=4")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import clip  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402

from . import util  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
torch.set_num_threads(1)


def get_embedding(dataset, dataset_name, embedding_length):
    
    with torch.no_grad():
        iteration = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)

        datax = torch.empty((len(dataset), embedding_length))
        datay = torch.empty((len(dataset)))

        element_index = 0
        for data in tqdm.tqdm(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            if dataset_name == "cub":
                image, labels, _ = data
            else:
                image, labels = data
            image = image.to(device)

            if dataset_name in ["cub", "cifar10", "cifar100"]:
                image_features = model.encode_image(image).float()
                image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)
            else:
                image_features = image.clone()


            datax[element_index : element_index + image_features.size(0)] = image_features.cpu().clone()
            datay[element_index : element_index + image_features.size(0)] = labels.cpu().clone()

            element_index = element_index + image_features.size(0)

            iteration += 1

    return datax.numpy(), datay.numpy()




def main(dataset):
    # dataset = dataset_name  # "cifar100" #"imagenet" #"places365" #"cub" #cifar10
    train_ds, test_ds = util.get_data(preprocess, dataset)
    module_dir = Path(__file__).parent

    concepts = util.get_concepts(module_dir / ("concept_sets/" + dataset + ".txt"))
    text = clip.tokenize(concepts).to(device)

    

    with torch.no_grad():
        text_features = model.encode_text(text)
        dictionary = text_features.T.float()

        dictionary = dictionary / torch.linalg.norm(dictionary, axis=0)
        print(dictionary.shape)

        torch.save(dictionary, module_dir /  f'saved_files/{dataset}_dictionary.pt')
        

        datax, datay = get_embedding(train_ds, dataset, dictionary.shape[0])

        np.save(module_dir / f"saved_files/{dataset}_train_embeddings.npy", datax)
        np.save(module_dir / f"saved_files/{dataset}_train_labels.npy", datay)

        datax_test, datay_test = get_embedding(test_ds, dataset, dictionary.shape[0])

        np.save(module_dir / f"saved_files/{dataset}_test_embeddings.npy", datax_test)
        np.save(module_dir / f"saved_files/{dataset}_test_labels.npy", datay_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dataset",
        "--dataset_name",
        type=str,
        choices=["imagenet", "places365", "cub", "cifar10", "cifar100"],
    )
    args = parser.parse_args()
    main(args.dataset_name)
