"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
"""

import json

from autora.variable import VariableCollection, Variable
from autora.experimentalist.random import pool
from autora.experiment_runner.firebase_prolific import firebase_runner
from autora.state import StandardState, on_state, Delta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sweetbean.sequence import Block, Experiment
from sweetbean.stimulus import TextStimulus

from trial_sequence import trial_sequences
from stimulus_sequence import stimulus_sequence
# *** Set up variables *** #
# independent variable is coherence in percent (0 - 100)
# dependent variable is rt in ms (0 - 10000)
variables = VariableCollection(
    independent_variables=[
        Variable(name="score_before", value_range=(0, 400)),
        Variable(name="score_after", value_range=(0, 400)),
        Variable(name="trial_index", allowed_values=np.linspace(0, 23, 24)),
        Variable(name="choice_index", allowed_values=np.linspace(0, 3, 4)),
        Variable(name="value", value_range=(-205, 205)),
        Variable(name="high_score", allowed_values=np.linspace(0, 300, 2)),
        ],
    dependent_variables=[Variable(name="type", value_range=(0, 1))])

# *** State *** #
# With the variables, we can set up a state. The state object represents the state of our
# closed loop experiment.


state = StandardState(
    variables=variables,
)

# *** Components/Agents *** #
# Components are functions that run on the state. The main components are:
# - theorist
# - experiment-runner
# - experimentalist
# See more about components here: https://autoresearch.github.io/autora/


# ** Theorist ** #
# Here we use a linear regression as theorist, but you can use other theorists included in
# autora (for a list: https://autoresearch.github.io/autora/theorist/)

theorist = LogisticRegression()


# To use the theorist on the state object, we wrap it with the on_state functionality and return a
# Delta object.
# Note: The if the input arguments of the theorist_on_state function are state-fields like
# experiment_data, variables, ... , then using this function on a state object will automatically
# use those state fields.
# The output of these functions is always a Delta object. The keyword argument in this case, tells
# the state object witch field to update.


@on_state()
def theorist_on_state(experiment_data, variables):
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    x = experiment_data[ivs]
    y = experiment_data[dvs]
    return Delta(models=[theorist.fit(x, y)])


# ** Experimentalist ** #
# Here, we use a random pool and use the wrapper to create a on state function
# Note: The argument num_samples is not a state field. Instead, we will pass it in when calling
# the function


@on_state()
def experimentalist_on_state(variables, num_samples):
    return Delta(conditions=pool(variables, num_samples))


# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key

firebase_credentials = {
  "type": "service_account",
  "project_id": "sweetbean-experiment",
  "private_key_id": "ceeb10d8b6603f71c46af9bceeafd67513f8d3f8",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDKFIQWmkT7DANw\niSo54e5ydwRBmQFmvh4AVRcBZGrO5KRV1WV6imcaS7hWYLCkNzXEeJnmK1FY3WOo\ni1dqNBDcM2LgoWFjaMoltyXzrNv9oFhrzfXUQxwqYWxfMTlArsjTUc9yWqqq4Ksd\nkp7Q9XA6yf0R6OQsj0Sr6gydcJd1cqtCEsIvVUvIdiZPlJPCZf0ZkZGxnFTQeOJk\n79c7vhj2SutJVY0lOYFGsy3fgaq7I4IcwsKeW+IRIIomoq43HXNCx/vajgkkTTsu\nprW9HVMOqud5ZoEOtAAsmj1cehvytdjH5X0UYlsGCb/s5hjt9G3KTFx4mgsL2xCs\n4X9m6rM5AgMBAAECggEAWGnQftkhPBt998Mzg96rRk53QMISQGMG2ktgRN4r8vhL\nRDiY7RvXz0P402X7cuWq8xp27yLdtP0r6KaKaO99vvIGnVROOUG0S/sNMkdbeuSo\nOwaCO0vfa1VzplRxgbDXMcnV8ujjSd8BTu+C59aysy4DuM3F4w8N0w+UVF/aDGqH\nlC3RqPrJVw5sYB8IwGeDes2vzSQl3s2E1/6pia7u5SWkxX8vhCtLwcwMwujM4zyX\nVqL6oUJQwuM+ueV3nJDPzAwsYSsHcp0afEkiwaO9n9Jnh5bZt8GpE5TH51ZEXfgB\nGaYwT2JRKEKMuDpnVb/JymhB+f/2s9kdT8w8pfJnbwKBgQDpyC+iaqVM+EWQCEbf\nI16utJXif21hZhuhrotK7mlVFdnzTF6TuQ31Nvck0Jbu5dccaCgazYOxy1HB8zTT\nxivz053l6aSBLzPRiRFtKbAjFWb54eBOaDmoRxdwUN/Ki9gOfBFF1f5QEU7bKYyN\nEdmRaGWNAzt2GEeCEXFrRxL+zwKBgQDdSQm556emMbxkj94VQ5bIKqCoLPTZ4a0N\nBekvBcpcaIAzZiI8PeSYm+zijmIHRViozFPl3lOPt35jTZmh1jKaTfu9leL1frwG\nGXSY6t0qcTJBmLMHD/XHxJpsw3rkyOzFXdR1DED+wm6F1SEGwhwR/U7neQfP78m9\nhCci8F3vdwKBgQDbHvJe5lynZzE0Tj23Whyd3c9663snRqBxi/stMYdy47dUTul6\nxoHprCo7zHMb2jwkeQ/WB3j/hZXNF1sVf/KkaF3gKH0zRH3qUPIPgnqAd8f8QRWQ\nCCq6ql+yu2r3GtpYwTsjXO8wNvjVfP0rIGbv3o0IfdYW26zyoczQA18viwKBgQC6\njf/QJfeVyipskUmGncO5nw5wme4W3gZ5izkqdnRC8arbKkjEht14t0O/QJBuVs1H\nCXPVwFisOeMortxNrvpcUlBgZcPAegbkEYPWA3NPe70FxklwM8lekGYsOaUayjWq\nss8RmrIU1TA+Tg8Y1n65v9dMmCG48QwgZRBliUV4QwKBgQCP8fytW534WGHH7qm3\nu50rjwjER8PReOAfAA3R53S8x5R4EDdGbdU0mOwjk0K+TMXpzsycKnibN0vvk8mk\nwDoi+xODFIWQQ+uRz1ZtMmfNruSOKVsEqaDB7B5lcGKzsw2kDvmLxkO/IzVarcZA\nZIJ70bZvyYQ1kpBQ5E+nIAOrAg==\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-ap5gl@sweetbean-experiment.iam.gserviceaccount.com",
  "client_id": "117772886570355590756",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-ap5gl%40sweetbean-experiment.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# simple experiment runner that runs the experiment on firebase
experiment_runner = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=100,
    sleep_time=5)


# Again, we need to wrap the runner to use it on the state. Here, we send the raw conditions.
@on_state()
def runner_on_state(conditions):
    res = []
    for idx, c in conditions.iterrows():
        i_1 = c['S1']
        i_2 = c['S2']
        # get a timeline via sweetPea
        timeline = trial_sequences(i_1, i_2, 10)[0]
        # get js code via sweetBeaan
        js_code = stimulus_sequence(timeline, i_1, i_2)
        res.append(js_code)
    
    conditions_to_send = conditions.copy()
    conditions_to_send['experiment_code'] = res
    # upload and run the experiment:
    data_raw = experiment_runner(conditions_to_send)

    # process the experiment data
    experiment_data = pd.DataFrame()
    for item in data_raw:
        _lst = json.loads(item)['trials']
        _df = trial_list_to_experiment_data(_lst)
        experiment_data = pd.concat([experiment_data, _df], axis=0)
    return Delta(experiment_data=experiment_data)

# def trial_list_to_experiment_data(trial_sequence):
#     """
#     Parse a trial sequence (from jsPsych) into dependent and independent variables
#     independent: S1, S2
#     dependent: rt
#     """
#     res_dict = {
#         'S1': [],
#         'S2': [],
#         'rt': []
#     }
#     for trial in trial_sequence:
#         # Filter trials that are not ROK (instructions, fixation, ...)
#         if trial['trial_type'] != 'rok':
#             continue
#         # Filter trials without rt
#         if 'rt' not in trial or trial['rt'] is None:
#             continue
#         # the intensity is equivalent to the number of oobs (set in sweetBean script)
#         # rt is a default value of every trial
#         s1 = trial['number_of_oobs'][0]
#         s2 = trial['number_of_oobs'][1]
#         rt = trial['rt']
        
#         res_dict['S1'].append(int(s1))
#         res_dict['S2'].append(int(s2))
#         res_dict['rt'].append(float(rt))
    
#     dataframe_raw = pd.DataFrame(res_dict)
    
#     # Calculate the mean rt for each S1/S2 combination
#     grouped = dataframe_raw.groupby(['S1', 'S2']).mean().reset_index()

#     return grouped

def trial_list_to_experiment_data(trial_sequence):
    """
    Parse a trial sequence (from jsPsych) into dependent and independent variables.
    Extracts the following:
    - Independent Variables: score_before, score_after, trial_index, choice_index, value
    - Dependent Variable: type
    """
    res_dict = {
        'score_before': [],
        'score_after': [],
        'trial_index': [],
        'choice_index': [],
        'value': [],
        'high_score': [],
        'type': []
    }
    
    for trial in trial_sequence:
        # Filter out trials that are not of the appropriate type (e.g., 'iowaGambling')
        if trial['trial_type'] != 'iowaGambling':
            continue
        
        # Extract the variables based on the provided keys
        score_before = trial.get('score_before')
        score_after = trial.get('score_after')
        trial_index = trial.get('trial_index')
        choice_index = trial.get('choice_index')
        value = trial.get('value')
        high_score = trial.get('high_score')
        trial_type = trial.get('type', 0)  # Default to 0 if 'type' is not available

        # Ensure all necessary data is present
        if None in (score_before, score_after, trial_index, choice_index, value, high_score):
            continue

        # Append data to the result dictionary
        res_dict['score_before'].append(float(score_before))
        res_dict['score_after'].append(float(score_after))
        res_dict['trial_index'].append(int(trial_index))
        res_dict['choice_index'].append(int(choice_index))
        res_dict['value'].append(float(value))
        res_dict['high_score'].append(int(high_score))
        res_dict['type'].append(int(trial_type))
    
    # Convert the result dictionary to a pandas DataFrame
    dataframe_raw = pd.DataFrame(res_dict)
    
    return dataframe_raw

# Now, we can run our components
for _ in range(3):
    state = experimentalist_on_state(state, num_samples=2)  # Collect 2 conditions per iteration
    state = runner_on_state(state)
    state = theorist_on_state(state)


# *** Report the data *** #
# If you changed the theorist, also change this part

def report_logistic_fit(m: LogisticRegression, feature_names=None, precision=4):
    # Coefficients (weights)
    coef = m.coef_.flatten()
    
    # Intercept
    intercept = m.intercept_.item()

    # Format feature names
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(len(coef))]

    # Log-odds equation
    terms = [f"{np.round(c, precision)} * {fn}" for c, fn in zip(coef, feature_names)]
    log_odds_eq = " + ".join(terms)
    s = f"log-odds = {log_odds_eq} + {np.round(intercept, precision)}"

    # Odds ratios (exponentiated coefficients)
    odds_ratios = np.exp(coef)
    odds_ratios_str = ", ".join([f"{fn}: {np.round(or_, precision)}" for fn, or_ in zip(feature_names, odds_ratios)])
    
    s += f"\nOdds Ratios: {odds_ratios_str}"
    
    return s


print(report_logistic_fit(state.models[0]))
print(report_logistic_fit(state.models[-1]))
