from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition", "Starts"),
        ("Gas", "Starts"),
        ("Starts", "Moves"),
        ("KeyPresent", "Starts")  # New node KeyPresent
    ]
)

# Defining the CPDs
cpd_battery = TabularCPD(variable="Battery", variable_card=2, values=[[0.70], [0.30]],
                         state_names={"Battery": ['Works', "Doesn't work"]})

cpd_gas = TabularCPD(variable="Gas", variable_card=2, values=[[0.40], [0.60]],
                     state_names={"Gas": ['Full', "Empty"]})

cpd_radio = TabularCPD(variable="Radio", variable_card=2,
                       values=[[0.75, 0.01], [0.25, 0.99]],
                       evidence=["Battery"], evidence_card=[2],
                       state_names={"Radio": ["turns on", "Doesn't turn on"], "Battery": ['Works', "Doesn't work"]})

cpd_ignition = TabularCPD(variable="Ignition", variable_card=2,
                          values=[[0.75, 0.01], [0.25, 0.99]],
                          evidence=["Battery"], evidence_card=[2],
                          state_names={"Ignition": ["Works", "Doesn't work"], "Battery": ['Works', "Doesn't work"]})

# Updated CPD for Starts considering KeyPresent
cpd_starts = TabularCPD(variable="Starts", variable_card=2,
                        values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                                [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
                        evidence=["Ignition", "Gas", "KeyPresent"], evidence_card=[2, 2, 2],
                        state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"],
                                     "Gas": ['Full', "Empty"], "KeyPresent": ['yes', 'no']})

cpd_moves = TabularCPD(variable="Moves", variable_card=2,
                       values=[[0.8, 0.01], [0.2, 0.99]],
                       evidence=["Starts"], evidence_card=[2],
                       state_names={"Moves": ["yes", "no"], "Starts": ['yes', 'no']})

# New CPD for KeyPresent
cpd_key_present = TabularCPD(variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
                             state_names={"KeyPresent": ['yes', 'no']})

# Associating the CPDs with the model
car_model.add_cpds(cpd_battery, cpd_gas, cpd_radio, cpd_ignition, cpd_starts, cpd_moves, cpd_key_present)

#  is valid
print("Model valid:", car_model.check_model())

# Queries
car_infer = VariableElimination(car_model)

# Given that the car will not move, what is the probability that the battery is not working?
result_1 = car_infer.query(variables=['Battery'], evidence={'Moves': 'no'})
print("Query 1 Result: \n", result_1)

# Given that the radio is not working, what is the probability that the car will not start?
result_2 = car_infer.query(variables=['Starts'], evidence={'Radio': "Doesn't turn on"})
print("\nQuery 2 Result: \n", result_2)

# Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
result_3 = car_infer.query(variables=['Radio'], evidence={'Battery': 'Works'})
result_3_with_gas = car_infer.query(variables=['Radio'], evidence={'Battery': 'Works', 'Gas': 'Full'})
print("\nQuery 3 Result without gas evidence: \n", result_3)
print("\nQuery 3 Result with gas evidence: \n", result_3_with_gas)

# Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?
result_4 = car_infer.query(variables=['Ignition'], evidence={'Moves': 'no'})
result_4_with_no_gas = car_infer.query(variables=['Ignition'], evidence={'Moves': 'no', 'Gas': 'Empty'})
print("\nQuery 4 Result without gas evidence: \n", result_4)
print("\nQuery 4 Result with gas evidence: \n", result_4_with_no_gas)

# What is the probability that the car starts if the radio works and it has gas in it?
result_5 = car_infer.query(variables=['Starts'], evidence={'Radio': 'turns on', 'Gas': 'Full'})
print("\nQuery 5 Result: \n", result_5)

