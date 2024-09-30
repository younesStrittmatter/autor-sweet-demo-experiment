"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
"""

import json
import os
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
variables = VariableCollection(
    independent_variables=[
        Variable(name="trial_index", allowed_values=np.arange(0, 24)),  # Trials index: 24 trials
        Variable(name="choice_index", allowed_values=np.arange(0, 4)),  # Choice: 4 choices (e.g., 4 card decks)
        Variable(name="payoff_reward_a", value_range=(0, 50)),  # Reward for choice A
        Variable(name="payoff_penalty_a", value_range=(-50, 0)),  # Penalty for choice A
        Variable(name="payoff_reward_b", value_range=(0, 50)),  # Reward for choice B
        Variable(name="payoff_penalty_b", value_range=(-50, 0)),  # Penalty for choice B
        Variable(name="payoff_reward_c", value_range=(0, 50)),  # Reward for choice C
        Variable(name="payoff_penalty_c", value_range=(-50, 0)),  # Penalty for choice C
        Variable(name="payoff_reward_d", value_range=(0, 50)),  # Reward for choice D
        Variable(name="payoff_penalty_d", value_range=(-50, 0)),  # Penalty for choice D
        Variable(name="win_chance", value_range=(0.4, 0.6)),  # Probability of winning on a trial
        Variable(name="starting_points", value_range=(100, 300)),  # Initial points assigned to the participant
    ],
    dependent_variables=[Variable(name="choice_outcome", value_range=(0, 1))]  # Outcome of choice (win or lose)
)

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
# Now, we will sample random combinations of the independent variables before running the experiment.

@on_state()
def experimentalist_on_state(variables):
    num_trials = 24
    trials_per_block = 6
    num_blocks = num_trials // trials_per_block

    independent_data = {
        "trial_index": np.arange(num_trials),
        "choice_index": np.random.randint(0, 4, size=num_trials),
        "payoff_reward_a": [],
        "payoff_penalty_a": [],
        "payoff_reward_b": [],
        "payoff_penalty_b": [],
        "payoff_reward_c": [],
        "payoff_penalty_c": [],
        "payoff_reward_d": [],
        "payoff_penalty_d": [],
        "win_chance": [],
        "starting_points": [200] * num_trials
    }

    for block in range(num_blocks):
        reward_a = np.random.randint(10, 30)
        penalty_a = np.random.randint(-30, 0)
        reward_b = np.random.randint(10, 30)
        penalty_b = np.random.randint(-30, 0)
        reward_c = np.random.randint(10, 30)
        penalty_c = np.random.randint(-30, 0)
        reward_d = np.random.randint(10, 30)
        penalty_d = np.random.randint(-30, 0)
        win_chance = np.random.uniform(0.4, 0.6)

        for trial in range(trials_per_block):
            index = block * trials_per_block + trial
            independent_data["payoff_reward_a"].append(reward_a)
            independent_data["payoff_penalty_a"].append(penalty_a)
            independent_data["payoff_reward_b"].append(reward_b)
            independent_data["payoff_penalty_b"].append(penalty_b)
            independent_data["payoff_reward_c"].append(reward_c)
            independent_data["payoff_penalty_c"].append(penalty_c)
            independent_data["payoff_reward_d"].append(reward_d)
            independent_data["payoff_penalty_d"].append(penalty_d)
            independent_data["win_chance"].append(win_chance)

    df = pd.DataFrame(independent_data)
    return Delta(conditions=df)




# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key

firebase_credentials = os.getenv('FIREBASE_API')

# simple experiment runner that runs the experiment on firebase
experiment_runner = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=100,
    sleep_time=5)


# Again, we need to wrap the runner to use it on the state. Here, we send the raw conditions.
@on_state()
def runner_on_state(conditions):
    # Convert the payoff scheme and other variables into the JavaScript code required for the experiment
    js_code = conditions.to_dict(orient="records")
    js_code = json.dumps(js_code)  # Convert to JSON format for JavaScript

    js_script = f"""
    <script>
        const TRIALS = 24;
        const START_POINTS = 200;
        let highest_score = 0;

        jsPsych.init({{
            timeline: timeline,
            display_element: 'jspsych-experiment',
            on_finish: function() {{
                jsPsych.data.displayData();
            }}
        }})

        let values = {js_code};  // Injected js_code
        let flips = [];
        let score = START_POINTS;

        timeline = [];
        for (let i = 0; i < TRIALS; i++) {{
            let flip = Math.random() < values[i]['win_chance'] ? 0 : 1;  // Simulate win/loss

            timeline.push({{
                type: jsPsychIowaGambling,
                values: [
                    [values[i]['payoff_reward_a'], values[i]['payoff_penalty_a']],
                    [values[i]['payoff_reward_b'], values[i]['payoff_penalty_b']],
                    [values[i]['payoff_reward_c'], values[i]['payoff_penalty_c']],
                    [values[i]['payoff_reward_d'], values[i]['payoff_penalty_d']]
                ],
                current_score: score,
                on_finish: (data) => {{
                    score += data.score_after;  // Update the score after each trial
                }}
            }});
        }}

        jsPsych.run(timeline);
    </script>
    """
    
    # Send the experiment script to firebase and run it
    conditions_to_send = pd.DataFrame({"experiment_code": [js_script]})
    data_raw = experiment_runner(conditions_to_send)

    # Process the experiment data
    experiment_data = pd.DataFrame()
    for item in data_raw:
        _lst = json.loads(item)["trials"]
        _df = trial_list_to_experiment_data(_lst)
        experiment_data = pd.concat([experiment_data, _df], axis=0)
    
    return Delta(experiment_data=experiment_data)


# Function to parse the experimental data
def trial_list_to_experiment_data(trial_sequence):
    res_dict = {
        'trial_index': [],
        'choice_index': [],
        'payoff_reward_a': [],
        'payoff_penalty_a': [],
        'payoff_reward_b': [],
        'payoff_penalty_b': [],
        'payoff_reward_c': [],
        'payoff_penalty_c': [],
        'payoff_reward_d': [],
        'payoff_penalty_d': [],
        'win_chance': [],
        'starting_points': [],
        'score_before': [],
        'score_after': [],
        'choice_outcome': []
    }

    for trial in trial_sequence:
        res_dict['trial_index'].append(trial.get('trial_index'))
        res_dict['choice_index'].append(trial.get('choice_index'))
        res_dict['payoff_reward_a'].append(trial.get('payoff_reward_a'))
        res_dict['payoff_penalty_a'].append(trial.get('payoff_penalty_a'))
        res_dict['payoff_reward_b'].append(trial.get('payoff_reward_b'))
        res_dict['payoff_penalty_b'].append(trial.get('payoff_penalty_b'))
        res_dict['payoff_reward_c'].append(trial.get('payoff_reward_c'))
        res_dict['payoff_penalty_c'].append(trial.get('payoff_penalty_c'))
        res_dict['payoff_reward_d'].append(trial.get('payoff_reward_d'))
        res_dict['payoff_penalty_d'].append(trial.get('payoff_penalty_d'))
        res_dict['win_chance'].append(trial.get('win_chance'))
        res_dict['starting_points'].append(trial.get('starting_points'))
        res_dict['score_before'].append(trial.get('score_before'))
        res_dict['score_after'].append(trial.get('score_after'))
        res_dict['choice_outcome'].append(trial.get('choice_outcome'))
    
    return pd.DataFrame(res_dict)


# Run the components in a loop
for _ in range(3):
    state = experimentalist_on_state(state)  # Collect independent variables
    state = runner_on_state(state)
    state = theorist_on_state(state)

# ** Report the Logistic Regression Fit ** #
def report_logistic_fit(m: LogisticRegression):
    print("Logistic Regression Coefficients:")
    print(m.coef_)
    print("Intercept:")
    print(m.intercept_)
    
# Report the theorist's model fit
report_logistic_fit(theorist)