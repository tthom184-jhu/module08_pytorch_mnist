
"""
run.py
Train a CNN on MNIST with PyTorch.

Usage:
    python run.py  # defaults: e=2, b=1000, l=0.001
    python run.py -e 5 -b 500 -l 0.01
"""


from utilities import MNISTTrainer, get_args

# run the main program from file execution
if __name__ == '__main__':

    # retrieve arguments from the user
    args = get_args()
    print(f"num_epochs={args.num_epochs}, batch_size={args.batch_size}, "
          f"learning_rate={args.learning_rate}")

    # Set the arguments for the trainer class
    trainer = MNISTTrainer(args.num_epochs, args.batch_size, args.learning_rate)
    
    # train the model
    trainer.main_run()
    