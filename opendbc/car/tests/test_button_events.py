from opendbc.car import ButtonSpec, create_button_events_from_specs, structs

ButtonType = structs.CarState.ButtonEvent.Type


def test_create_button_events_from_specs_multiplexed_signal():
  buttons = (
    ButtonSpec(ButtonType.accelCruise, "SCM_BUTTONS", "CRUISE_BUTTONS", (4,), ("resume", "increase")),
    ButtonSpec(ButtonType.decelCruise, "SCM_BUTTONS", "CRUISE_BUTTONS", (3,), ("set", "decrease")),
  )
  button_states = {button: False for button in buttons}

  vl = {"SCM_BUTTONS": {"CRUISE_BUTTONS": 0}}
  assert create_button_events_from_specs(vl, button_states, buttons) == []

  vl["SCM_BUTTONS"]["CRUISE_BUTTONS"] = 4
  events = create_button_events_from_specs(vl, button_states, buttons)
  assert [(event.type, event.pressed) for event in events] == [(ButtonType.accelCruise, True)]

  vl["SCM_BUTTONS"]["CRUISE_BUTTONS"] = 3
  events = create_button_events_from_specs(vl, button_states, buttons)
  assert [(event.type, event.pressed) for event in events] == [
    (ButtonType.accelCruise, False),
    (ButtonType.decelCruise, True),
  ]

  vl["SCM_BUTTONS"]["CRUISE_BUTTONS"] = 0
  events = create_button_events_from_specs(vl, button_states, buttons)
  assert [(event.type, event.pressed) for event in events] == [(ButtonType.decelCruise, False)]


def test_create_button_events_from_specs_independent_signals():
  buttons = (
    ButtonSpec(ButtonType.setCruise, "GRA_ACC_01", "GRA_Tip_Setzen", (1,), ("set",)),
    ButtonSpec(ButtonType.accelCruise, "GRA_ACC_01", "GRA_Tip_Hoch", (1,), ("increase",)),
  )
  button_states = {button: False for button in buttons}

  vl = {"GRA_ACC_01": {"GRA_Tip_Setzen": 0, "GRA_Tip_Hoch": 0}}
  assert create_button_events_from_specs(vl, button_states, buttons) == []

  vl["GRA_ACC_01"]["GRA_Tip_Setzen"] = 1
  events = create_button_events_from_specs(vl, button_states, buttons)
  assert [(event.type, event.pressed) for event in events] == [(ButtonType.setCruise, True)]

  vl["GRA_ACC_01"]["GRA_Tip_Hoch"] = 1
  events = create_button_events_from_specs(vl, button_states, buttons)
  assert [(event.type, event.pressed) for event in events] == [(ButtonType.accelCruise, True)]

  vl["GRA_ACC_01"]["GRA_Tip_Setzen"] = 0
  vl["GRA_ACC_01"]["GRA_Tip_Hoch"] = 0
  events = create_button_events_from_specs(vl, button_states, buttons)
  assert [(event.type, event.pressed) for event in events] == [
    (ButtonType.setCruise, False),
    (ButtonType.accelCruise, False),
  ]
