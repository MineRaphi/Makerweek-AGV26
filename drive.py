import requests
import numpy as np

AGV_BASE_URL = "http://172.17.1.42/api/agv"
WHEEL_DIAMETER = 95 # 95mm

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
