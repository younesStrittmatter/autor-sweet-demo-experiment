"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
"""

import json

from autora.variable import VariableCollection, Variable
from autora.experimentalist.random import random_sample
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
def experimentalist_on_state(variables):
    a = np.array(random_sample(
    [[[10, 0], [5, 0], [5, 0], [10, 0]], [[10, -25], [5, -5], [5, -5], [10, -25]], [[25, -200], [5, -5], [5, -5], [25, -200]]]))
    # Combine all lists into one
    combined_list = [item for sublist in a for item in sublist]
    df = pd.DataFrame({'payoff_scheme': [combined_list]})
    return Delta(conditions=df)


# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key

firebase_credentials = {
  "type": "service_account",
  "project_id": "sweetbean-experiment",
  "private_key_id": "71c0102fb21c5a630ab92f7b1921ec71a2bfcb31",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDnVrWGOebjLNW/\ns+ZHpgEymAj6fBWCGfIw+tY2jq/be21r6yRbo9b4x8qLDybSacmR2hy5CFfSoGpt\nt7mxzyzvj//Ac0CM9BA012g7sRC/2d79vWrLYCdkHJdr/IITSTchUlI0kbOFdML+\nundN94oWYLMb3Al/t329O5FH6dmbJFBG5wvZ8o9pmy6QRqSFZ+G7MJyIxQ67Xzwn\nrC5zhVQkGgmgr677CIQcQRNksKbKWncmatuMlrpjXz1Qejvd4ofhpJeIMhSbN0JZ\nx1u5ExJmk8+Im3gG7tNxiYi1fKeR17YapbcTpLBwY3PEYeFABn/K2TH/vjavwxZ/\nZBG22aG/AgMBAAECggEAMvt198XM72XTGXNhHYslGmxNFj7AUrK7IDY3fUCG2vzh\niXYBxjxeROdq6KiHKrnrlTwXNmPzTtTRf6qMvvXkdksq1tPPdwDWjX6nVvhXs6Al\nN5BF99oR9Eskx8SXOf7ZqntE6JbvlEq+KnsXjdQu122qK0qbwUzD5i85fjq4HStB\nN3io6h1WkJ2z0mRM3474LLiZOlPwEfDgahqDo8YcAYKAC79GQZv8QcpZnLdTeioz\nkXjc9mRzombUKsjZL9s869yfFiKWQf1oTa9hAFp98x9DNFBl2Pmu/kkZAZqxOwBV\nol7wUcvhqpOvKWfhoz8W2LSneswlcO+fUpD+ERxHgQKBgQD9gG6zwtrWrpvyzbQm\n2CuQAPMxWhpbohYfRq8kwDtfJkt5XkwbXU1/9ke3f12brQstThj+bcCaldaJNCyk\nWzBjBvkzDdTKNKF/C+nCzmYMMVmGr0ZpwwVpgpQ/3JzFUpr6BcHpyfWulOfDGr9T\n4gbJCc6kbhnJxsDk1eo9IUBWfwKBgQDpnlxnJIGCWfdotWaHUk9XveCl0UD/0Gbm\n6DQmJMSQTaX7QoZscj9ICdoIMqi0RZGMdfT6U5QHHWBQ9vBXh3BISM1WJJT5otZ0\nEovSB+nDDiiJRMB7sR91GcMVJikE+bXa5IIyV4XuOlKXPl0I2pwCjJWi+PeRhzy3\nhEiY5ymUwQKBgQD7T/bveS5QhHwQIsQFemr9cSOneocE7tR1nzKFAZoagzFxmf1j\nZ4UsZbDFhpv7eHrLKFB4879swT0VekcDjW+TzNcCOSUKbVDpTZsqSEo8rjPt5Reu\nQ+u6pPxpr0EwEeuYEFskddZ9hBubfYnOFBbb+UAGHSytr7+NXVDB15Qb6wKBgAzi\n5lfuJJKrIcGN2AT43lWJrL2YyEwUE8kC3/WGq60GC3TLm5yZxLHVkUhIexPOjpO/\n4e54875cuXZd2K4LU385PNJWnD0U5V1rtHi2ZQeUXVoNB80K3SBZdnBRNYwHtidH\n2YKrX0DfyLR9BSa64EYnuQ1PTGCjpA6/Zj3A6oNBAoGBAMKYChj8aYQ+W4xhTJlx\nYw7327kkrC5cC629J53slarnJZJuHDF/5RymssmdLJq2cBMppS3cc17pY9BaAKP3\nUeKXqOV9ydzfUPaFHDQeUXSUO0mAsR4oztBINWNBs0hMuohKrobbigbJA5ZZwll9\njEWRXIoY6+oA9a483p890qjP\n-----END PRIVATE KEY-----\n",
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
    # for idx, c in conditions.iterrows():
    #     i_1 = c['S1']
    #     i_2 = c['S2']
    #     # get a timeline via sweetPea
    #     timeline = trial_sequences(i_1, i_2, 10)[0]
    #     # get js code via sweetBeaan
    #     js_code = stimulus_sequence(timeline, i_1, i_2)
    
    
    js_code = conditions['payoff_scheme']
    
    js_code = json.dumps(js_code.tolist())

    # JavaScript code template where js_code is injected
    js_script = f"""
    <script>
        const experiment_payoff = 1;
        const TRIALS = 24;
        const WIN_CHANCE = .5;
        const START_POINTS = 200;
        let highest_score = 0; // Initialize high_score with a default value

        <!-- jsPsych = initJsPsych({{
            on_finish: function () {{
                jsPsych.data.get().filter({{trial_type: 'iowaGambling'}});
            }}
        }}) -->

        jsPsych.init({{
            timeline: timeline, // `timeline` will be defined by the injected js_code
            display_element: 'jspsych-experiment', // The div where the experiment will run
            on_finish: function() {{
                jsPsych.data.displayData(); // Callback to handle the data
            }}
        }})

        if (Math.random() >= 0.5) {{
            highest_score = START_POINTS * 1.5; // Update high_score if the condition is met
        }}

        let values = {js_code};  // Injected js_code
        let flips = [];
        let winChance = WIN_CHANCE;
        let score = START_POINTS;

        timeline = [];
        for (let i = 0; i < TRIALS; i++) {{
            if (experiment_payoff === 2) {{
                winChance = 1 - ((i / TRIALS) * 0.8); // Decreases linearly over time
            }}

            if (Math.random() < winChance) {{
                flip = 0; // Win
            }} else {{
                flip = 1; // Loss
            }}

            timeline.push({{
                type: jsPsychIowaGambling,
                values: [values[0][flip], values[1][flip], values[2][flip], values[3][flip]],
                reward_penalty: [
                    [values[0][0], values[0][1]],
                    [values[1][0], values[1][1]],
                    [values[2][0], values[2][1]],
                    [values[3][0], values[3][1]]
                ],
                chance: winChance,
                current_score: () => {{
                    return score;
                }},
                high_score: () => {{
                    return highest_score;
                }},
                on_finish: (data) => {{
                    score = data.score_after;
                    if (data.reward === 5) {{
                        data['type'] = 0;
                    }} else {{
                        data['type'] = 1;
                    }}
                }}
            }});
        }}

        jsPsych.run(timeline);

        function shuffle(array) {{
            let currentIndex = array.length, randomIndex;

            while (currentIndex > 0) {{
                randomIndex = Math.floor(Math.random() * currentIndex);
                currentIndex--;

                [array[currentIndex], array[randomIndex]] = [
                    array[randomIndex], array[currentIndex]];
            }}

            return array;
        }}
    </script>
    """
    
    
    conditions_to_send = pd.DataFrame()
    conditions_to_send['experiment_code'] = js_script
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
    state = experimentalist_on_state(state)  # Collect 1 payoff scheme per iteration
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
