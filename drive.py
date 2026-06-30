import requests
import numpy as np
import random

AGV_BASE_URL = "http://172.17.1.42/api/agv"

# distance in mm
WHEEL_DIAMETER = 95
WHEEL_DISTANCE = 210 

# steps for one rotation
MAX_STEPS = 3550
STEPS_PER_DEGREE = 10

def enable_wheels():
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/enable",
        json={ "stepper": "on" }
    )
    return response

def disable_wheels():
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/enable",
        json={ "stepper": "off" }
    )
    return response

def set_velocity(left, right):
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/setVelocity",
        json={
            "velLeft_perc": left,
            "velRight_perc": right
        }
    )
    return response

def stop():
    return set_velocity(0.0, 0.0)

def move_steps(left, right):
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/setMoveRelative",
        json={
            "leftDelta_steps": left,
            "rightDelta_steps": right
        }
    )
    return response

def set_step(left, right):
    response = requests.post(
        f"{AGV_BASE_URL}/stepper/setMoveAbsolute",
        json={
            "leftPos_steps": left,
            "rightPos_steps": right
        }
    )
    return response

def angletostep(angle):
    steps = angle * STEPS_PER_DEGREE
    return int(round(steps))

def rotate(angle):
    move_steps(-angletostep(angle), angletostep(angle))

if __name__ == "__main__":
    enable_wheels()
    rotate(random.randint(60, 250))