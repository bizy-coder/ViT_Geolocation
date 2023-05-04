from trained_model_experiments_util import *


def evaluate(model_name, model_num, save_path, augment=False):
    # Create directory to save results
    os.makedirs(save_path, exist_ok=True)

    learn = create_learner(model_name, augment=augment, test_set=True)
    if type(model_num) == int:
        learn = load_wandb_model(model_num, learn)
    else:
        file = torch.load(model_num, map_location=device)
        learn.model.load_state_dict(file)
    learn.show_results(1)

    # Print size of dataset
    print(f"Size of dataset: {len(learn.dls.valid_ds)}")
    # Print number of classes
    print(f"Number of classes: {len(learn.dls.vocab)}")

    preds, targs = get_predictions(learn)
    preds, targs = preds.numpy(), targs.numpy()
    # Save to numpy array
    np.save(f"{save_path}/preds.npy", preds)
    np.save(f"{save_path}/targs.npy", targs)

    states = learn.dls.vocab
    cm = display_confusion_matrix(
        preds, targs, states, save_path=f"{save_path}/confusion_matrix.png"
    )

    print(get_most_confused(cm, states))

    top_accs = plot_top_k_accuracy(
        preds, targs, save_path=f"{save_path}/top_k_accuracy.png"
    )

    print(top_accs)

    plot_class_precision_and_recall(
        cm, states, save_path=f"{save_path}/precision_recall.png"
    )

    plot_class_f1(cm, states, save_path=f"{save_path}/f1.png")


if __name__ == "__main__":
    evaluate(
        model_name="vit_small_patch32_224",
        model_num=16,
        save_path="vit_with_aug_20_augment",
        augment=True,
    )
