# fuzzy_fever.py

import re
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def assess_fever_description(fever_desc: str) -> str:
    """
    Interpret a user's fever description (e.g. "mild", "moderate", "high")
    and (optionally) a heart rate from the user's text using fuzzy logic.
    Returns a recommended response based on both temperature and heart rate.
    """

    # 1. Define fuzzy variables
    # Temperature from 36.0 to 42.0 in steps of 0.1
    temperature = ctrl.Antecedent(np.arange(36.0, 42.0, 0.1), 'temperature')
    # Heart rate from 40 to 180 in steps of 1 (example range)
    heart_rate  = ctrl.Antecedent(np.arange(40, 181, 1), 'heart_rate')
    concern     = ctrl.Consequent(np.arange(0, 101, 1), 'concern')

    # 2. Define membership functions for temperature
    temperature['normal']   = fuzz.trimf(temperature.universe, [36.0, 36.0, 37.0])
    temperature['mild']     = fuzz.trimf(temperature.universe, [37.0, 37.5, 38.0])
    temperature['moderate'] = fuzz.trimf(temperature.universe, [38.0, 39.0, 40.0])
    temperature['high']     = fuzz.trimf(temperature.universe, [39.5, 41.0, 42.0])

    # 3. Define membership functions for heart rate
    #    You can adjust these ranges to be more precise if needed
    heart_rate['normal']   = fuzz.trapmf(heart_rate.universe, [40, 60, 80, 90])
    heart_rate['elevated'] = fuzz.trimf(heart_rate.universe, [80, 100, 120])
    heart_rate['high']     = fuzz.trapmf(heart_rate.universe, [110, 130, 180, 180])

    # 4. Define membership functions for concern level (0-100)
    concern['low']    = fuzz.trimf(concern.universe, [0, 0, 25])
    concern['medium'] = fuzz.trimf(concern.universe, [25, 50, 75])
    concern['high']   = fuzz.trimf(concern.universe, [75, 100, 100])

    # 5. Define fuzzy rules (combine temperature & heart_rate)
    #    Adjust rules as makes sense for your scenario:
    rule1 = ctrl.Rule(temperature['normal'] & heart_rate['normal'],   concern['low'])
    rule2 = ctrl.Rule(temperature['normal'] & heart_rate['elevated'], concern['medium'])
    rule3 = ctrl.Rule(temperature['mild']   & heart_rate['normal'],   concern['medium'])
    rule4 = ctrl.Rule(temperature['mild']   & heart_rate['elevated'], concern['medium'])
    rule5 = ctrl.Rule(temperature['moderate'] & heart_rate['normal'], concern['medium'])
    rule6 = ctrl.Rule(temperature['moderate'] & heart_rate['high'],   concern['high'])
    rule7 = ctrl.Rule(temperature['high'] | heart_rate['high'],       concern['high'])

    # 6. Create and simulate the control system
    fever_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    fever_sim = ctrl.ControlSystemSimulation(fever_ctrl)

    # 7. Map textual input to numeric temperature
    user_input_temp = 36.5  # default if no fever keyword found
    desc_lower = fever_desc.lower()

    if "mild" in desc_lower:
        user_input_temp = 37.5
    elif "moderate" in desc_lower:
        user_input_temp = 38.5
    elif any(word in desc_lower for word in ["high", "severe"]):
        user_input_temp = 40.0

    # 8. Attempt to parse a numeric heart rate from the description (if provided).
    #    If none is found, we default to a "normal" 80 BPM.
    user_input_hr = 80
    # Simple regex to look for a number (like 90 or 120, etc.)
    hr_match = re.search(r'\b(\d{2,3})\b', desc_lower)
    if hr_match:
        # Convert the captured text to int. You could refine the detection 
        # by searching specifically for "heart rate" patterns, if needed.
        parsed_value = int(hr_match.group(1))
        # Only accept plausible heart-rate ranges for humans
        if 40 <= parsed_value <= 180:
            user_input_hr = parsed_value

    # 9. Run the fuzzy system with both temperature & heart_rate
    fever_sim.input['temperature'] = user_input_temp
    fever_sim.input['heart_rate']  = user_input_hr
    fever_sim.compute()

    concern_level = fever_sim.output['concern']

    # 10. Convert numeric concern to a user-facing message
    if concern_level < 30:
        msg = (
            f"Temp ~{user_input_temp}°C, HR ~{user_input_hr} bpm => Concern: {concern_level:.1f}%.\n"
            "Advice: Looks quite low risk. Continue to monitor and stay hydrated."
        )
    elif concern_level < 70:
        msg = (
            f"Temp ~{user_input_temp}°C, HR ~{user_input_hr} bpm => Concern: {concern_level:.1f}%.\n"
            "Advice: Mild to moderate concern. Rest, drink fluids, and monitor both temperature and heart rate."
        )
    else:
        msg = (
            f"Temp ~{user_input_temp}°C, HR ~{user_input_hr} bpm => Concern: {concern_level:.1f}%.\n"
            "Advice: High concern. Please consult a healthcare provider."
        )

    return msg
