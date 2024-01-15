#Aqsa Noreen
import os

def get_output_of_file(filename):
    return os.popen(f'python {filename}').read()

if __name__ == "__main__":
    # Get output for nltk_intro.py
    HMM_output = get_output_of_file("HMM.py")
    print("The outputs for HMM.py:")
    print(HMM_output)

    alarm_output = get_output_of_file("alarm.py")
    print("The outputs for alarm.py:")
    print(alarm_output)

    carnet_output = get_output_of_file("carnet.py")
    print("The outputs for carnet.py:")
    print(carnet_output)
