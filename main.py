import os
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parkinsons_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ParkinsonsDetection")

def setup_environment():
    """Set up the environment and required directories"""
    from config import DATA_PATH, SPIRAL_DATASET_PATH, WAVE_DATASET_PATH, MODEL_SAVE_PATH

    # Create necessary directories
    for directory in [DATA_PATH, SPIRAL_DATASET_PATH, WAVE_DATASET_PATH, MODEL_SAVE_PATH]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created (if it didn't exist): {directory}")

def prepare_data(args):
    """Prepare the dataset for training and testing"""
    logger.info("Preparing data...")
    from prepare_data import prepare_sample_data, process_raw_data, preprocess_images, create_clinical_data_file

    if args.create_sample:
        logger.info(f"Creating sample data with {args.num_samples} samples per class...")
        prepare_sample_data(data_dir=args.data_dir, num_samples=args.num_samples, overwrite=args.overwrite)

    if args.process_raw:
        logger.info(f"Processing raw data from {args.raw_dir}...")
        process_raw_data(raw_dir=args.raw_dir, processed_dir=args.data_dir)

    if args.preprocess:
        logger.info("Preprocessing images...")
        preprocess_images(data_dir=args.data_dir)

    if args.create_clinical:
        logger.info("Creating clinical data file...")
        create_clinical_data_file(data_dir=args.data_dir, overwrite=args.overwrite)

    logger.info("Data preparation completed.")

def train_models(args):
    """Train the selected models"""
    logger.info("Training models...")
    import train

    # Construct the command-line arguments for the train module
    train_args = argparse.Namespace(
        model=args.model,
        train_all=args.train_all,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_dir,
        output_dir=args.output_dir,
        transfer_model=args.transfer_model
    )

    # Run the training
    train.main(train_args)
    logger.info("Training completed.")

def evaluate_models(args):
    """Evaluate trained models"""
    logger.info("Evaluating models...")
    import evaluate

    # Construct the command-line arguments for the evaluate module
    eval_args = argparse.Namespace(
        model_dir=args.model_dir,
        data_path=args.data_dir,
        plot=args.plot,
        save_results=args.save_results,
        output_dir=args.output_dir
    )

    # Run the evaluation
    evaluate.main(eval_args)
    logger.info("Evaluation completed.")

def predict(args):
    """Make predictions using trained models"""
    logger.info("Making predictions...")
    import predict as predict_module

    if args.web_interface:
        logger.info("Launching web interface...")
        # Call the main function from predict.py with command-line arguments
        args_dict = {
            "model_dir": args.model_dir,
            "web_interface": True
        }
        predict_args = argparse.Namespace(**args_dict)
        predict_module.main(predict_args)
    elif args.image_path:
        logger.info(f"Predicting on image: {args.image_path}")
        # Call the main function from predict.py with command-line arguments
        args_dict = {
            "model_dir": args.model_dir,
            "image_path": args.image_path,
            "model": args.model,
            "web_interface": False
        }
        predict_args = argparse.Namespace(**args_dict)
        predict_module.main(predict_args)
    else:
        logger.error("No prediction method specified. Use --image_path or --web_interface.")

def run_tests(args):
    """Run unit tests"""
    logger.info("Running tests...")
    import unittest
    import test_suite

    # Run all tests
    unittest.main(module=test_suite, argv=['first-arg-is-ignored'], exit=False)
    logger.info("Tests completed.")

def main():
    """Main entry point of the program"""
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parkinson's Disease Detection System")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Setup parser
    setup_parser = subparsers.add_parser("setup", help="Set up environment")

    # Prepare data parser
    prepare_parser = subparsers.add_parser("prepare", help="Prepare data")
    prepare_parser.add_argument("--data_dir", type=str, default="data", help="Base directory for data")
    prepare_parser.add_argument("--create_sample", action="store_true", help="Create sample data for testing")
    prepare_parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to create per class")
    prepare_parser.add_argument("--process_raw", action="store_true", help="Process raw data into the proper directory structure")
    prepare_parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing raw data files")
    prepare_parser.add_argument("--preprocess", action="store_true", help="Preprocess images (resize, grayscale, threshold)")
    prepare_parser.add_argument("--create_clinical", action="store_true", help="Create synthetic clinical data file")
    prepare_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data")

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--model", type=str, default="cnn", help="Model type to train (cnn, rf, svm, gb, transfer, hybrid)")
    train_parser.add_argument("--train_all", action="store_true", help="Train all models")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--data_dir", type=str, default="data", help="Base directory for data")
    train_parser.add_argument("--output_dir", type=str, default="saved_models", help="Directory to save trained models")
    train_parser.add_argument("--transfer_model", type=str, default="vgg16", help="Base model for transfer learning (vgg16, resnet50)")

    # Evaluate parser
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    evaluate_parser.add_argument("--model_dir", type=str, default="saved_models", help="Directory containing trained models")
    evaluate_parser.add_argument("--data_dir", type=str, default="data", help="Base directory for data")
    evaluate_parser.add_argument("--plot", action="store_true", help="Generate and display plots")
    evaluate_parser.add_argument("--save_results", action="store_true", help="Save evaluation results to CSV")
    evaluate_parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results and plots")

    # Predict parser
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--model_dir", type=str, default="saved_models", help="Directory containing trained models")
    predict_parser.add_argument("--image_path", type=str, help="Path to image for prediction")
    predict_parser.add_argument("--model", type=str, default="ensemble", help="Model to use for prediction (cnn, rf, svm, gb, transfer_vgg16, transfer_resnet50, or ensemble)")
    predict_parser.add_argument("--web_interface", action="store_true", help="Launch web interface for interactive prediction")

    # Test parser
    test_parser = subparsers.add_parser("test", help="Run tests")

    # Demo parser
    demo_parser = subparsers.add_parser("demo", help="Run demo notebook")

    # Parse arguments
    args = parser.parse_args()

    # Handle the command
    if args.command == "setup":
        setup_environment()
    elif args.command == "prepare":
        prepare_data(args)
    elif args.command == "train":
        train_models(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "test":
        run_tests(args)
    elif args.command == "demo":
        try:
            import jupyter
            logger.info("Opening demo notebook...")
            import subprocess
            subprocess.run(["jupyter", "notebook", "demo.ipynb"])
        except ImportError:
            logger.error("Jupyter notebook is not installed. Please install it with: pip install notebook")
    else:
        parser.print_help()

def init_command():
    """Function to be called when the script is executed from the command line"""
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    init_command()
