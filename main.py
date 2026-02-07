import argparse

from DroneControl.drone_control import DroneControl


def main():
    parser = argparse.ArgumentParser(description="Drone Control via Hand gestures")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Only Video feedback no Drone connection , debug purposes",
    )
    args = parser.parse_args()

    controller = DroneControl(debug=args.debug)
    controller.run()


if __name__ == "__main__":
    main()