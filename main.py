from argparse import ArgumentParser

import Add_duration_to_all_data
import eval_passengers_up
import eval_trip_duration

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        help="Specify the task: 'passenger_boardings' or 'trip_duration'")
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()


    if args.task == 'passenger_boardings':
        print("Running Task 1: Predicting Passenger Boardings")
        eval_passengers_up.run(args.training_set, args.test_set, args.out)
    elif args.task == 'trip_duration':
        print("Running Task 2: Predicting Trip Duration")
        Add_duration_to_all_data.create_duration_column(args.training_set)
        eval_trip_duration.run("data/train_bus_schedule_duration.csv", args.test_set, args.out)

    else:
        print(f"Unknown task: {args.task}. Please specify either 'passenger_boardings' or 'trip_duration'.")