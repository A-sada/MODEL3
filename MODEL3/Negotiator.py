from Strategy_Vehicle_ver1 import Vehicle
from negmas import SAOMechanism
from negmas.utilities import UtilityFunction
from negmas.outcomes import Outcome

def Nego1(vehicleA,vehicleB,NegA,NegB):
    ListA = []
    outcomes = [{"taskA": taskA, "taskB": taskB} for taskA in vehicleA.tasks for taskB in vehicleB.tasks + [None]]

    mechanism = SAOMechanism(
        outcomes=outcomes,
        n_steps=10
    )
    mechanism.add(NegA)
    mechanism.add(NegB)
    result = mechanism.run()
    return result


