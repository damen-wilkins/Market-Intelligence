from scripts.run_stage2_route_compatibility_diagnostic import main as run_diagnostic
from scripts.run_stage2_route_aware_multiclass import main as run_route_aware


def main():
    run_diagnostic()
    run_route_aware()


if __name__ == "__main__":
    main()
