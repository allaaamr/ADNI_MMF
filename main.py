from data.dataset import *
from data.utils import *
from models.ResNet3D import *
from models.utils import *
from utils.utils import *
from utils.stat_utils import *
import argparse
from explainability.explainability import run_explainability_from_ckpt, run_explainability
from timeit import default_timer as timer
from argparse import Namespace
from torch.utils.data import DataLoader, WeightedRandomSampler
from explainability.GradCam import *
from explainability.explainability import run_gradcam_from_pt


def main(args):

    if args.xai:

        png_path, pred = run_gradcam_from_pt( model_ctor=lambda: ResNet3D(num_classes=3, in_channels=1, layers=(2,2,2,2), widths=(32,64,128,256)),
            checkpoint_path=args.xai_ckpt or "/home/alaa.mohamed/ADNI_MMF/best_radio.pth", mri_pt_path=args.xai_pt,                         # add this arg below
            target_module=None,                              
            clinical=1.0,                                    
            out_png=args.xai_out_png,                   
            plane=args.xai_plane,                             #
            slice_idx=args.xai_slice,                       
            strategy="cammax" if args.xai_cammax else "middle",
            class_idx=args.xai_class,
            device="cuda"
        )
                
         

    else:
        args.target_size = (128, 128, 128)
        fold_metrics = []   # list of dicts (each fold's acc/auroc)
        accs = []           # list of accuracy values (float)
        aucs = []           # list of AUROC values (float)
        best_fold = {"score": -1.0, "fold_idx": None, "dir": None}

        for fold_idx in range (1,3):
            print(f"\n========== Training on FOLD {fold_idx} ==========")


            train_csv = pd.read_csv(os.path.join(args.splits_path, f"fold_{fold_idx}", f"train_fold_{fold_idx}.csv"))
            val_csv   = pd.read_csv(os.path.join(args.splits_path, f"fold_{fold_idx}", f"val_fold_{fold_idx}.csv"))
            train_ids, train_labels = train_csv.PTID.values, train_csv.DIAGNOSIS.values
            val_ids, val_labels     = val_csv.PTID.values,  val_csv.DIAGNOSIS.values

            train_ds = MMFDataset(args, ptids=train_ids, labels=train_labels,images_dir=args.images_dir, train=True)
            val_ds = MMFDataset(args, ptids=val_ids, labels=val_labels,images_dir=args.images_dir, train=False, scaler=train_ds.scaler)


            nw = getattr(args, "num_workers", 1)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,num_workers=nw, pin_memory=True,persistent_workers=bool(nw))
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=nw, pin_memory=True,persistent_workers=bool(nw), prefetch_factor=2 if nw else None)

            
            torch.set_num_threads(1)  # avoid CPU oversubscription in workers
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("device ", device)
            model = train((train_loader, val_loader), args, device)


            logits, y_true = collect_logits_and_targets(model, val_loader, device)
            metrics = compute_multiclass_metrics(logits, y_true)
            print(f"[FOLD {fold_idx}] ACC={metrics['acc']:.3f} | AUROC(macro)={metrics['auroc_macro']:.3f}")

            # Save fold-level artifacts
            fold_dir = os.path.join(getattr(args, "splits_dir", "cv_splits"), f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            # Save predictions and ground truth for later statistical testing
            np.save(os.path.join(fold_dir, "val_logits.npy"), logits)
            np.save(os.path.join(fold_dir, "val_y_true.npy"), y_true)

            # Save metrics JSON for bookkeeping
            save_json(metrics, os.path.join(fold_dir, "metrics.json"))

            # Store metrics in memory for statistical testing across folds
            fold_metrics.append(metrics)
            accs.append(metrics["acc"])
            aucs.append(metrics["auroc_macro"])

            
            score = metrics["auroc_macro"]
            if not np.isfinite(score):   # if AUROC fails (NaN), fall back to accuracy
                score = metrics["acc"]
            if score > best_fold["score"]:
                best_fold = {
                    "score": float(score),
                    "fold_idx": fold_idx,
                    "dir": fold_dir
                }
            
        # After all folds (cross-validation summary)
        aucs = np.array(aucs, dtype=float)

        # Compute mean ± std ± CI across folds
        acc_mean, acc_std, acc_ci = mean_ci(accs)
        auc_mean, auc_std, auc_ci = mean_ci(aucs)

        print("\n===== Cross-fold summary =====")
        print(f"ACC  = {acc_mean:.3f} ± {acc_std:.3f}  (95% CI: {acc_ci[0]:.3f}, {acc_ci[1]:.3f})")
        print(f"AUROC= {auc_mean:.3f} ± {auc_std:.3f}  (95% CI: {auc_ci[0]:.3f}, {auc_ci[1]:.3f})")

        # One-sample t-test vs chance (baseline random classifier: 1/3 for 3-class problem)
        chance_acc = 1.0/3.0
        t_stat, p_val = stats.ttest_1samp(accs, popmean=chance_acc, alternative="greater")
        print(f"T-test vs chance (ACC > {chance_acc:.3f}): t={t_stat:.3f}, p={p_val:.6f}")

        fold_dir = os.path.join(getattr(args, "splits_dir", "cv_splits"), f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        ckpt = {
            "model": model.state_dict(),
            "arch": "ResNet3D",          # or args.mode/your arch name
            "epoch": int(getattr(args, "epoch", -1)),
            "target_size": tuple(args.target_size),
            "num_classes": 3,
            "extras": {"fold_idx": fold_idx}
        }
        torch.save(ckpt, os.path.join(fold_dir, "model_best.pth"))

    end = timer()
    print('Time: %f seconds' % ( end - start))


parser = argparse.ArgumentParser()
parser.add_argument("--xai", action="store_true")
parser.add_argument("--xai_technique", choices=["grad","shap"], default="grad")
parser.add_argument("--xai_save_nifti", action="store_true")
parser.add_argument("--xai_out", type=str, default="best_models")
parser.add_argument("--xai_class", type=int, default=2)
parser.add_argument("--xai_max", type=int, default=8)
parser.add_argument("--xai_ckpt", type=str, default=None, help="Path to best .pth checkpoint")


parser.add_argument('--seed', 			 type=int, default=42, help='Random seed for reproducible experiment (default: 1)')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--epochs',      type=int, default=15, help='Maximum  number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--loss',        	 type=str, choices=[ 'cox', 'nll'], default='nll')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-5, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

parser.add_argument('--images_dir',  type=str, default="/home/alaa.mohamed/ADNI_data/Raw/Images", help='Path to the folder containing mri scans')
parser.add_argument('--csv_file',  type=str, default="/home/alaa.mohamed/ADNI_MMF/data/clinical.csv", help='Path to clinical csv')
parser.add_argument('--splits_path',  type=str, default="/home/alaa.mohamed/ADNI_MMF/cv_splits/", help='Path to clinical csv')
parser.add_argument('--id_col',  type=str, default="PTID", help='Patient ID column name in csv file')
parser.add_argument('--label_col',  type=str, default="DIAGNOSIS", help='Label column name in csv file')
parser.add_argument('--val_size',  type=float, default=0.2)

parser.add_argument('--mode', 			 type=str,choices=['radio', 'clinical', 'late_fusion', 'hybrid'], default='radio', help='Model to run')



args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Sets Seed for reproducible experiments.
def seed_torch(seed=42):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)



if __name__ == "__main__":
    start = timer()
    main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))



