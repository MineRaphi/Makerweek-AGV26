import requests
import random
from config import *


def enable_wheels():
    """Powers on the stepper motors so the AGV can move."""
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/enable",
        json={ "stepper": "on" }
    )
    return response


def disable_wheels():
    """Powers off the stepper motors (AGV won't respond to movement commands)."""
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/enable",
        json={ "stepper": "off" }
    )
    return response


def set_velocity(left, right):
    """
    Sets continuous wheel speed as a percentage (not a fixed move — runs until changed).
    left / right: velocity percentage for each wheel (e.g. -100 to 100)
    """
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/setVelocity",
        json={
            "velLeft_perc": left,
            "velRight_perc": right
        }
    )
    return response


def stop():
    """Stops both wheels immediately by setting velocity to 0."""
    return set_velocity(0.0, 0.0)


def move_steps(left, right):
    """
    Moves each wheel by a RELATIVE number of motor steps from its current position.
    Positive = forward, negative = backward, for that wheel.
    """
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/setMoveRelative",
        json={
            "leftDelta_steps": left,
            "rightDelta_steps": right
        }
    )
    return response


def set_step(left, right):
    """
    Moves each wheel to an ABSOLUTE step position (not relative to current position).
    Useful for resetting to a known state.
    """
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/setMoveAbsolute",
        json={
            "leftPos_steps": left,
            "rightPos_steps": right
        }
    )
    return response


def angletostep(angle):
    """Converts a rotation angle (degrees) into the equivalent number of motor steps."""
    steps = angle * STEPS_PER_DEGREE
    return int(round(steps))


def rotate(angle):
    """
    Rotates the AGV in place by the given angle (degrees).
    Wheels move in opposite directions (left back, right forward, or vice versa)
    so the AGV pivots without driving forward.
    Positive angle = turn one way, negative = turn the other (direction depends on wiring).
    """
    move_steps(-angletostep(angle), angletostep(angle))


def move_mm(distance):
    """
    Drives the AGV straight forward (or backward, if negative) by the given
    distance in millimeters. Both wheels move the same number of steps.
    """
    move_steps(int(distance * STEPS_PER_MM), int(distance * STEPS_PER_MM))


# --- Quick manual test: power on and spin a random amount when run directly ---
if __name__ == "__main__":
    enable_wheels()
    rotate(random.randint(60, 250))