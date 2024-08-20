(function (jspsych) {
    "use strict";
    const info = {
        name: 'iowaGambling',
        description: '',
        parameters: {
            values: {
                type: jspsych.ParameterType.ARRAY,
                pretty_name: 'Values for the cards',
                default: [100, 100, 50, 50],
                description: 'The rewards for the decks.'
            },
            deck_colors: {
                type: jspsych.ParameterType.ARRAY,
                pretty_name: 'Deck color',
                default: ['grey', 'grey', 'grey', 'grey'],
                description: 'Deck color'
            },
            current_score: {
                type: jspsych.ParameterType.INT,
                pretty_name: 'Current points',
                default: 2000,
                description: 'Current points'
            },
            deck_names: {
                type: jspsych.ParameterType.ARRAY,
                pretty_name: 'Deck names',
                default: ['A', 'B', 'C', 'D'],
                description: 'Deck names'
            },
            reward_penalty: {
                type: jspsych.ParameterType.ARRAY,
                pretty_name: 'Values for the cards',
                default: [[100, -250], [100, -250], [50, -50], [50, -50]],
                description: 'The rewards for the decks.'
            },
            chance: {
                type: jspsych.ParameterType.FLOAT,
                pretty_name: 'Values for the cards',
                default: .5,
                description: 'The rewards for the decks.'
            },
            high_score: {
                type: jspsych.ParameterType.INT,
                pretty_name: 'High Score',
                default: 0,
                description: 'High score based on random value and current score'
            }
        }
    }

    class jsPsychIowaGambling {
        constructor(jsPsych) {
            this.jsPsych = jsPsych;
        }

        trial(display_element, trial) {
            let style = document.createElement('style');
            style.appendChild(document.createTextNode(css_style));

            display_element.appendChild(style)
            let pointDiv = document.createElement('div')
            pointDiv.className += ' score'
            pointDiv.innerText = trial.current_score;
            display_element.appendChild(pointDiv)

            // Check if high_score is greater than zero, then create and append it
            if (trial.high_score > 0) {
                let highScoreDiv = document.createElement('div');
                highScoreDiv.className += ' high-score';
                highScoreDiv.innerText = `High Score: ${trial.high_score}`;
                display_element.appendChild(highScoreDiv);
            }

            let decks = []
            for (let i = 0; i < 4; i++) {
                let deck = document.createElement('div')
                deck.className += ' card-container'

                deck.style.background = trial.deck_colors[i]
                if (i < 2) {
                    deck.style.top = '10vh'
                } else {
                    deck.style.bottom = '10vh'
                }
                if (i === 0 || i === 2) {
                    deck.style.left = '10vw'
                } else {
                    deck.style.right = '10vw'
                }
                display_element.appendChild(deck)
                let inner = document.createElement('div')
                inner.className += ' card-inner'
                inner.id = `iowa-gambling-${i}`
                deck.appendChild(inner)
                let front = document.createElement('div')
                front.className += ' card-front'
                front.innerText = trial.values[i]
                inner.appendChild(front)
                let back = document.createElement('div')
                back.className += ' card-back'
                back.id = `iowa-gambling-back-${i}`
                back.innerText = trial.deck_names[i]
                back.style.background = trial.deck_colors[i]
                inner.appendChild(back)
            }
            let responded = false
            const after_response = (i) => {
                if (responded) {
                    return
                }
                responded = true
                pointDiv.innerText = trial.current_score + trial.values[i]
                display_element.querySelector("#iowa-gambling-" + i).style.transform = 'rotateY(180deg)'
                let choice = ['A', 'B', 'C', 'D'][i]
                let value = trial.values[i]
                let trial_data = {
                    choice: choice,
                    value: value,
                    score_before: trial.current_score,
                    score_after: trial.current_score + trial.values[i],
                    values: JSON.stringify(trial.values),
                    choice_index: i,
                    reward: trial.reward_penalty[i][0],
                    penalty: trial.reward_penalty[i][1],
                    chance: trial.chance
                };
                setTimeout(() => {
                    display_element.querySelector("#iowa-gambling-back-" + i).
                        style.visibility = 'hidden'
                },
                    200)
                setTimeout(() => {
                    display_element.innerHTML = ''
                    this.jsPsych.finishTrial(trial_data)
                }, 1000)
            };


            // Start RDK after delay


            // Set up the event listener for responses

            for (let i = 0; i < 4; i++) {
                display_element
                    .querySelector("#iowa-gambling-" + i)
                    .addEventListener("click", (e) => {
                        after_response(i);
                    });
            }
        }
    }

    jsPsychIowaGambling.info = info;

    window.jsPsychIowaGambling = jsPsychIowaGambling
})(jsPsychModule);


const css_style =
    `
    * {
    margin: 0;
    padding: 0;
    }
    
    .score {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: black;
    }

    .high-score {
        position: fixed;
        top: calc(50% + 30px);
        left: 50%;
        transform: translate(-50%, -50%);
        color: black;
    }
    
    .card-deck {
    width: 10vw;
    height: 15vw;
    border: 2px solid #fff;
    border-radius: 20px;
    position: fixed;
    cursor: pointer;
    }
    
    .card-container {
    position: fixed;
    width: 10vw;
    height: 15vw;
    outline: solid 1px black;
    border-radius: 1vw;
    perspective: 1000px;
    }
    
    .card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    text-align: center;
    transition: transform 0.4s;
    transform-style: preserve-3d;
    }
    
    .card-front, .card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    -moz-backface-visibility: hidden;
    -webkit-backface-visibility: hidden; /* Safari */
    backface-visibility: hidden;
    }
    
    .card-front {
    background-color: white;
    outline: solid 1px black;
    border-radius: 1vw;
    line-height: 15vw;
    color: black;

    transform: rotateY(180deg);
    }
    
    .card-back {
    color: white;
    outline: solid 1px black;
     line-height: 15vw;
    border-radius: 1vw;
    `

