from pathlib import Path
import traceback
import argparse

from experiments.exp_biclass import Exp_BiClass
from experiments.exp_extrap import Exp_Extrap


parser = argparse.ArgumentParser(
    description="Run all experiments within the Leit framework.")

# Args for the training process
parser.add_argument("--random-state", type=int, default=1, help="Random seed")
parser.add_argument("--proj-path", type=str,
                    default=str(Path(__file__).parents[0]))
parser.add_argument("--test-info", default="testing")
parser.add_argument("--leit-model", default="ivp_vae")
parser.add_argument("--num-dl-workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--epochs-min", type=int, default=1)
parser.add_argument("--epochs-max", type=int, default=1000,
                    help="Max training epochs")
parser.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience")
parser.add_argument("--weight-decay", type=float,
                    default=0.0001, help="Weight decay (regularization)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--lr-scheduler-step", type=int, default=20,
                    help="Every how many steps to perform lr decay")
parser.add_argument("--lr-decay", type=float, default=0.5,
                    help="Multiplicative lr decay factor")
parser.add_argument("--clip-gradient", action='store_false')
parser.add_argument("--clip", type=float, default=1)
parser.add_argument("--log-tool", default="logging",
                    choices=["logging", "wandb"])

# Args for datasets
parser.add_argument("--data", default="p12", help="Dataset name",
                    choices=["m4_full", "p12", "eicu"])
parser.add_argument("--num-samples", type=int, default=-1)
parser.add_argument("--variable-num", type=int,
                    default=37, choices=[96, 37, 41, 14])
parser.add_argument("--ts-full", action='store_true')
parser.add_argument("--del-std5", action='store_true')
parser.add_argument("--time-scale", default="time_max",
                    choices=["time_max", "self_max", "constant", "none", "max"])
parser.add_argument("--time-constant", type=float, default=2880)
parser.add_argument("--first-dim", default="batch",
                    choices=["batch", "time_series"])
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--t-offset", type=float, default=0.1)
parser.add_argument("--ml-task", default="biclass",
                    choices=["biclass", "extrap"])
parser.add_argument("--extrap-full", action='store_true')
parser.add_argument("--down-times", type=int, default=1,
                    help="downsampling timestamps")
parser.add_argument("--time-max", type=int, default=1439)
parser.add_argument("--next-start", type=float, default=1440)
parser.add_argument("--next-end", type=float, default=2880)
parser.add_argument("--next-headn", type=int, default=0)
parser.add_argument('--mask-drop-rate', type=float, default=0.0)
parser.add_argument("--norm", action='store_false')

# Args for IVP solvers
parser.add_argument("--ivp-solver", default="resnetflow",
                    choices=["resnetflow", "couplingflow", "gruflow", "ode"])
parser.add_argument("--hidden-layers", type=int, default=3,
                    help="Number of hidden layers")
parser.add_argument("--hidden-dim", type=int, default=128,
                    help="Size of hidden layer")
parser.add_argument("--activation", type=str, default="ELU",
                    help="Hidden layer activation")
parser.add_argument("--final-activation", type=str,
                    default="Tanh", help="Last layer activation")
parser.add_argument("--odenet", type=str, default="concat",
                    help="Type of ODE network", choices=["concat", "gru"])  # gru only in GOB
parser.add_argument("--ode_solver", type=str, default="dopri5",
                    help="ODE solver", choices=["dopri5", "rk4", "euler"])
parser.add_argument("--solver_step", type=float,
                    default=0.05, help="Fixed solver step")
parser.add_argument("--atol", type=float, default=1e-4,
                    help="Absolute tolerance")
parser.add_argument("--rtol", type=float, default=1e-3,
                    help="Relative tolerance")
parser.add_argument("--flow-layers", type=int, default=2,
                    help="Number of flow layers")
parser.add_argument("--time-net", type=str, default="TimeTanh", help="Name of time net",
                    choices=["TimeFourier", "TimeFourierBounded", "TimeLinear", "TimeTanh"])
parser.add_argument("--time-hidden-dim", type=int, default=8,
                    help="Number of time features (only for Fourier)")

# Args for VAE
parser.add_argument("--k-iwae", type=int, default=3)
parser.add_argument("--kl-coef", type=float, default=1.0)
parser.add_argument("--latent-dim", type=int, default=20)
parser.add_argument("--classifier-input", default="z0")
parser.add_argument("--train-w-reconstr", action='store_false')
parser.add_argument("--ratio-ce", type=float, default=1000)
parser.add_argument("--ratio-nl", type=float, default=1)
parser.add_argument("--ratio-zz", type=float, default=0)
parser.add_argument("--prior-mu", type=float, default=0.0)
parser.add_argument("--prior-std", type=float, default=1.0)
parser.add_argument("--obsrv-std", type=float, default=0.01)
parser.add_argument("--combine-methods", default="average",
                    choices=["average", "kl_weighted"])


if __name__ == "__main__":
    args = parser.parse_args()
    if args.ml_task == 'extrap':
        experiment = Exp_Extrap(args)
    elif args.ml_task == 'biclass':
        experiment = Exp_BiClass(args)
    else:
        raise ValueError("Unknown")

    try:
        experiment.run()
        experiment.finish()
    except Exception:
        with open(experiment.proj_path/"log"/"err_{}.log".format(experiment.args.exp_name), "w") as fout:
            print(traceback.format_exc(), file=fout)
