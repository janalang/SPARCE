import os
import torch
import random
import argparse
from datetime import datetime
from counterfactual_gan import CounterfactualTimeGAN
import pandas as pd
from dataloader import *


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20) 
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="motionsense", help="motionsense, simulated")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002) 
    parser.add_argument("--save_indicator", type=bool, default=False, help="False or True")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight of adversarial loss")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Weight of classification loss")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Weight of similarity loss")
    parser.add_argument("--lambda4", type=float, default=1.0, help="Weight of sparsity loss")
    parser.add_argument("--lambda5", type=float, default=1.0, help="Weight of jerk loss")
    parser.add_argument("--freeze_features", type=list, default=[])
    parser.add_argument("--seed", type=int, default=123, help='random seed for splitting data')
    parser.add_argument("--num_reps", type=int, default=1, help='number of repetitions of experiments')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--init_lambda", type=float, default=1.0)
    parser.add_argument("--approach", type=str, default="sparce")
    parser.add_argument("--save", type=bool, default=False, help="save experiment file, originals and cfs")
    parser.add_argument("--max_lambda_steps", type=int, default=5)
    parser.add_argument("--lambda_increase", type=float, default=0.001)

    return parser

def main():
    parser = init_argparse()
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    random.seed(args.seed)
    seeds = random.sample(range(0, 1000), args.num_reps)
    print(seeds)

    testdoc = []

    for rep in range(args.num_reps):

        args.seed = seeds[rep]
        print(f'----------------------------- Repetition {rep} / {args.num_reps}, Seed: {args.seed} -----------------------------')


        X_train_generator_input, train_dl_real_samples, train_dl_generator_input, test_dl_real_samples, test_dl_generator_input, train_max_samples, train_max_batches, test_max_samples, test_max_batches = prepare_counterfactual_data(args)

        # model and training parameters
        num_features = X_train_generator_input.shape[2]

        model = CounterfactualTimeGAN()
        model.build_model(args=args, device=device, num_features=num_features, bidirectional=True, hidden_dim_generator=256, layer_dim_generator=2, hidden_dim_discriminator=16, layer_dim_discriminator=1, classifier_model_name="bidirectional_lstm_classifier")
        model.train(train_dl=train_dl_real_samples, generator_dl = train_dl_generator_input, max_samples=train_max_samples, max_batches=train_max_batches)
        testdoc = model.test(test_dl=test_dl_real_samples, generator_dl = test_dl_generator_input, max_samples=test_max_samples, max_batches=test_max_batches, testdoc=testdoc)


    now = datetime.now()
    date_and_time = now.strftime("%d_%m_%Y_%H_%M_%S_")
    doc_df = pd.DataFrame(testdoc)
    print(doc_df)
    if args.save:
        doc_df.to_csv(os.path.join(os.getcwd(), 'experiments', args.dataset, date_and_time + args.approach + '.csv'), index=False)
        print("Experiment done. File saved.")

if __name__ == "__main__":
    main()